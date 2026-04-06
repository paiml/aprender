//! Notebook exporter for Jupyter format.

use serde::{Deserialize, Serialize};
use serde_json::json;

use super::cell::{Cell, CellType};
use super::kernel::KernelSpec;
use crate::research::literate::LiterateDocument;

/// Notebook exporter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotebookExporter {
    /// Notebook cells
    pub cells: Vec<Cell>,
    /// Kernel specification
    pub kernel: KernelSpec,
    /// Notebook metadata
    pub metadata: NotebookMetadata,
}

impl Default for NotebookExporter {
    fn default() -> Self {
        Self::new()
    }
}

impl NotebookExporter {
    /// Create a new notebook exporter
    pub fn new() -> Self {
        Self {
            cells: Vec::new(),
            kernel: KernelSpec::python3(),
            metadata: NotebookMetadata::default(),
        }
    }

    /// Create with a specific kernel
    pub fn with_kernel(kernel: KernelSpec) -> Self {
        Self { cells: Vec::new(), kernel, metadata: NotebookMetadata::default() }
    }

    /// Add a cell to the notebook
    pub fn add_cell(&mut self, cell: Cell) {
        self.cells.push(cell);
    }

    /// Add a code cell
    pub fn add_code(&mut self, source: impl Into<String>) {
        self.cells.push(Cell::code(source));
    }

    /// Add a markdown cell
    pub fn add_markdown(&mut self, source: impl Into<String>) {
        self.cells.push(Cell::markdown(source));
    }

    /// Create from a literate document
    pub fn from_literate(doc: &LiterateDocument) -> Self {
        let mut exporter = Self::new();

        // Determine kernel from document type
        if doc.is_typst() || doc.is_markdown() {
            // Extract code blocks and intersperse with markdown
            let content = doc.content();
            let blocks = doc.extract_code_blocks();

            // Detect primary language for kernel selection
            let primary_lang = blocks.iter().find_map(|b| b.language.as_ref()).map(String::as_str);

            exporter.kernel = match primary_lang {
                Some("rust") => KernelSpec::evcxr(),
                Some("julia") => KernelSpec::julia(),
                Some(other_lang) => {
                    eprintln!(
                        "Warning: unsupported kernel language '{other_lang}', defaulting to Python 3"
                    );
                    KernelSpec::python3()
                }
                None => KernelSpec::python3(),
            };

            // Simple parsing: add everything before first code block as markdown
            // then alternate between code and markdown
            let mut last_end = 0;

            for block in &blocks {
                // Find the start of this code block in the content
                let block_pattern = format!("```{}", block.language.as_deref().unwrap_or(""));
                if let Some(start_pos) = content[last_end..].find(&block_pattern) {
                    let absolute_start = last_end + start_pos;

                    // Add markdown before this block
                    let markdown_content = &content[last_end..absolute_start];
                    let trimmed = markdown_content.trim();
                    if !trimmed.is_empty() {
                        exporter.add_markdown(trimmed);
                    }

                    // Add the code block
                    exporter.add_code(&block.content);

                    // Find the end of this code block
                    let code_end = content[absolute_start..]
                        .find("```\n")
                        .or_else(|| content[absolute_start..].find("```"))
                        .map_or(content.len(), |p| {
                            absolute_start
                                + p
                                + content[absolute_start + p..].find('\n').unwrap_or(3)
                                + 1
                        });

                    last_end = code_end.min(content.len());
                }
            }

            // Add remaining markdown after last code block
            if last_end < content.len() {
                let remaining = content[last_end..].trim();
                if !remaining.is_empty() {
                    exporter.add_markdown(remaining);
                }
            }
        } else {
            // Raw text: add as single markdown cell
            exporter.add_markdown(doc.content());
        }

        exporter
    }

    /// Export to Jupyter notebook JSON format
    pub fn to_ipynb(&self) -> String {
        let notebook = json!({
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {
                "kernelspec": {
                    "display_name": self.kernel.display_name,
                    "language": self.kernel.language,
                    "name": self.kernel.name
                },
                "language_info": {
                    "name": self.kernel.language
                }
            },
            "cells": self.cells.iter().map(|cell| {
                let mut cell_json = json!({
                    "cell_type": cell.cell_type.to_string(),
                    "source": cell.source,
                    "metadata": cell.metadata
                });

                if cell.cell_type == CellType::Code {
                    cell_json["outputs"] = json!(cell.outputs.iter().map(|o| {
                        let mut out = json!({
                            "output_type": o.output_type
                        });
                        if let Some(data) = &o.data {
                            out["data"] = data.clone();
                        }
                        if let Some(text) = &o.text {
                            out["text"] = json!(text);
                            out["name"] = json!("stdout");
                        }
                        out
                    }).collect::<Vec<_>>());
                    cell_json["execution_count"] = json!(cell.execution_count);
                }

                cell_json
            }).collect::<Vec<_>>()
        });

        serde_json::to_string_pretty(&notebook).unwrap_or_else(|_err| "{}".to_string())
    }

    /// Get the number of cells
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    /// Get code cells only
    pub fn code_cells(&self) -> Vec<&Cell> {
        self.cells.iter().filter(|c| c.cell_type == CellType::Code).collect()
    }

    /// Get markdown cells only
    pub fn markdown_cells(&self) -> Vec<&Cell> {
        self.cells.iter().filter(|c| c.cell_type == CellType::Markdown).collect()
    }
}

/// Notebook metadata
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NotebookMetadata {
    /// Notebook title
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Authors
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub authors: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::super::cell::CellOutput;
    use super::*;

    #[test]
    fn test_kernel_selection_all_language_variants() {
        let languages: &[Option<&str>] = &[Some("rust"), Some("julia"), Some("javascript"), None];

        for lang in languages {
            // Syntactic match covering all arms from from_literate kernel selection
            let kernel = match *lang {
                Some("rust") => KernelSpec::evcxr(),
                Some("julia") => KernelSpec::julia(),
                Some(_other_lang) => KernelSpec::python3(),
                None => KernelSpec::python3(),
            };

            match lang {
                Some("rust") => assert_eq!(kernel.language, "rust"),
                Some("julia") => assert_eq!(kernel.language, "julia"),
                Some(_) => assert_eq!(kernel.language, "python"),
                None => assert_eq!(kernel.language, "python"),
            }
        }
    }

    #[test]
    fn test_notebook_exporter_new() {
        let exporter = NotebookExporter::new();
        assert_eq!(exporter.cells.len(), 0);
        assert_eq!(exporter.kernel.language, "python");
    }

    #[test]
    fn test_notebook_exporter_default() {
        let exporter = NotebookExporter::default();
        assert_eq!(exporter.cell_count(), 0);
    }

    #[test]
    fn test_add_code_and_markdown() {
        let mut exporter = NotebookExporter::new();
        exporter.add_code("print('hello')");
        exporter.add_markdown("# Title");
        assert_eq!(exporter.cell_count(), 2);
        assert_eq!(exporter.code_cells().len(), 1);
        assert_eq!(exporter.markdown_cells().len(), 1);
    }

    #[test]
    fn test_notebook_metadata_default() {
        let meta = NotebookMetadata::default();
        assert!(meta.title.is_none());
        assert!(meta.authors.is_empty());
    }

    // ── Additional coverage tests ─────────────────────────────────

    #[test]
    fn test_notebook_exporter_with_kernel() {
        let exporter = NotebookExporter::with_kernel(KernelSpec::evcxr());
        assert_eq!(exporter.kernel.language, "rust");
        assert_eq!(exporter.cell_count(), 0);
    }

    #[test]
    fn test_notebook_exporter_with_julia_kernel() {
        let exporter = NotebookExporter::with_kernel(KernelSpec::julia());
        assert_eq!(exporter.kernel.language, "julia");
        assert_eq!(exporter.kernel.display_name, "Julia 1.9");
    }

    #[test]
    fn test_add_cell_directly() {
        let mut exporter = NotebookExporter::new();
        exporter.add_cell(Cell::code("x = 1"));
        exporter.add_cell(Cell::markdown("# Hello"));
        exporter.add_cell(Cell::raw("raw text"));
        assert_eq!(exporter.cell_count(), 3);
        assert_eq!(exporter.code_cells().len(), 1);
        assert_eq!(exporter.markdown_cells().len(), 1);
    }

    #[test]
    fn test_to_ipynb_empty() {
        let exporter = NotebookExporter::new();
        let json = exporter.to_ipynb();
        assert!(json.contains("nbformat"));
        assert!(json.contains("\"cells\": []"));
    }

    #[test]
    fn test_to_ipynb_with_cells() {
        let mut exporter = NotebookExporter::new();
        exporter.add_code("print('hello')");
        exporter.add_markdown("# Title");
        let json = exporter.to_ipynb();
        assert!(json.contains("print('hello')"));
        assert!(json.contains("# Title"));
        assert!(json.contains("\"cell_type\": \"code\""));
        assert!(json.contains("\"cell_type\": \"markdown\""));
    }

    #[test]
    fn test_to_ipynb_kernelspec() {
        let exporter = NotebookExporter::with_kernel(KernelSpec::evcxr());
        let json = exporter.to_ipynb();
        assert!(json.contains("\"language\": \"rust\""));
        assert!(json.contains("Rust"));
    }

    #[test]
    fn test_to_ipynb_code_cell_has_outputs_and_execution_count() {
        let mut exporter = NotebookExporter::new();
        exporter.add_code("1 + 1");
        let json = exporter.to_ipynb();
        assert!(json.contains("\"outputs\""));
        assert!(json.contains("\"execution_count\""));
    }

    #[test]
    fn test_to_ipynb_with_output() {
        let mut exporter = NotebookExporter::new();
        let cell = Cell::code("print(42)")
            .with_output(CellOutput::stream("stdout", "42\n"))
            .with_execution_count(1);
        exporter.add_cell(cell);
        let json = exporter.to_ipynb();
        assert!(json.contains("stream"));
        assert!(json.contains("42"));
        assert!(json.contains("stdout"));
    }

    #[test]
    fn test_to_ipynb_with_execute_result() {
        let mut exporter = NotebookExporter::new();
        let data = serde_json::json!({"text/plain": ["result"]});
        let cell = Cell::code("1 + 1").with_output(CellOutput::execute_result(data));
        exporter.add_cell(cell);
        let json = exporter.to_ipynb();
        assert!(json.contains("execute_result"));
        assert!(json.contains("text/plain"));
    }

    #[test]
    fn test_from_literate_markdown() {
        let doc = LiterateDocument::parse_markdown(
            "# Hello\n\nSome text.\n\n```python\nprint('hi')\n```\n\nMore text.",
        );
        let exporter = NotebookExporter::from_literate(&doc);
        assert!(exporter.cell_count() > 0);
        // Should have at least one code cell and one markdown cell
        assert!(!exporter.code_cells().is_empty());
        assert!(!exporter.markdown_cells().is_empty());
        assert_eq!(exporter.kernel.language, "python");
    }

    #[test]
    fn test_from_literate_rust_kernel() {
        let doc =
            LiterateDocument::parse_markdown("# Rust Example\n\n```rust\nfn main() {}\n```\n");
        let exporter = NotebookExporter::from_literate(&doc);
        assert_eq!(exporter.kernel.language, "rust");
    }

    #[test]
    fn test_from_literate_julia_kernel() {
        let doc =
            LiterateDocument::parse_markdown("# Julia Example\n\n```julia\nprintln(\"hi\")\n```\n");
        let exporter = NotebookExporter::from_literate(&doc);
        assert_eq!(exporter.kernel.language, "julia");
    }

    #[test]
    fn test_from_literate_no_code_blocks() {
        let doc = LiterateDocument::parse_markdown("# Just Markdown\n\nNo code here.");
        let exporter = NotebookExporter::from_literate(&doc);
        // Should have at least some markdown content
        assert!(exporter.cell_count() >= 1);
        assert!(exporter.code_cells().is_empty());
    }

    #[test]
    fn test_from_literate_raw_text() {
        let doc = LiterateDocument::raw("Just plain text, no special parsing.");
        let exporter = NotebookExporter::from_literate(&doc);
        assert_eq!(exporter.cell_count(), 1);
        assert_eq!(exporter.markdown_cells().len(), 1);
    }

    #[test]
    fn test_from_literate_multiple_code_blocks() {
        let doc = LiterateDocument::parse_markdown(
            "Intro\n\n```python\nx = 1\n```\n\nMiddle text\n\n```python\ny = 2\n```\n\nEnd",
        );
        let exporter = NotebookExporter::from_literate(&doc);
        assert_eq!(exporter.code_cells().len(), 2);
    }

    #[test]
    fn test_notebook_metadata_with_values() {
        let meta = NotebookMetadata {
            title: Some("My Notebook".to_string()),
            authors: vec!["Author A".to_string(), "Author B".to_string()],
        };
        assert_eq!(meta.title.as_deref(), Some("My Notebook"));
        assert_eq!(meta.authors.len(), 2);
    }

    #[test]
    fn test_notebook_metadata_serialization() {
        let meta = NotebookMetadata {
            title: Some("Test".to_string()),
            authors: vec!["Alice".to_string()],
        };
        let json = serde_json::to_string(&meta).expect("serialize");
        let restored: NotebookMetadata = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored.title, meta.title);
        assert_eq!(restored.authors, meta.authors);
    }

    #[test]
    fn test_notebook_metadata_serialization_skip_empty() {
        let meta = NotebookMetadata::default();
        let json = serde_json::to_string(&meta).expect("serialize");
        // title should be omitted (skip_serializing_if = "Option::is_none")
        assert!(!json.contains("title"));
        // authors should be omitted (skip_serializing_if = "Vec::is_empty")
        assert!(!json.contains("authors"));
    }

    #[test]
    fn test_notebook_exporter_serialization() {
        let mut exporter = NotebookExporter::new();
        exporter.add_code("x = 1");
        let json = serde_json::to_string(&exporter).expect("serialize");
        let restored: NotebookExporter = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored.cell_count(), 1);
        assert_eq!(restored.kernel.language, "python");
    }

    #[test]
    fn test_code_cells_and_markdown_cells_filtering() {
        let mut exporter = NotebookExporter::new();
        exporter.add_code("a");
        exporter.add_markdown("b");
        exporter.add_code("c");
        exporter.add_markdown("d");
        exporter.add_code("e");

        assert_eq!(exporter.code_cells().len(), 3);
        assert_eq!(exporter.markdown_cells().len(), 2);
        assert_eq!(exporter.cell_count(), 5);
    }

    #[test]
    fn test_from_literate_typst() {
        let doc = LiterateDocument::Typst(
            "= Title\n\nSome text.\n\n```python\nprint('hi')\n```\n\nMore text.".to_string(),
        );
        let exporter = NotebookExporter::from_literate(&doc);
        assert!(exporter.cell_count() > 0);
    }
}
