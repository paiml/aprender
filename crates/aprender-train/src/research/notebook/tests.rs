//! Tests for notebook module.

use super::*;
use crate::research::literate::LiterateDocument;

#[test]
fn test_basic_notebook_export() {
    let mut exporter = NotebookExporter::new();
    exporter.add_markdown("# Hello World");
    exporter.add_code("print('Hello!')");

    let ipynb = exporter.to_ipynb();

    assert!(ipynb.contains("nbformat"));
    assert!(ipynb.contains("Hello World"));
    assert!(ipynb.contains("print('Hello!')"));
}

#[test]
fn test_code_cells_extracted() {
    let mut exporter = NotebookExporter::new();
    exporter.add_code("x = 1");
    exporter.add_markdown("Some text");
    exporter.add_code("y = 2");

    let code_cells = exporter.code_cells();
    assert_eq!(code_cells.len(), 2);
}

#[test]
fn test_markdown_cells_preserved() {
    let mut exporter = NotebookExporter::new();
    exporter.add_markdown("# Title");
    exporter.add_markdown("## Subtitle");
    exporter.add_code("code");

    let md_cells = exporter.markdown_cells();
    assert_eq!(md_cells.len(), 2);
}

#[test]
fn test_evcxr_kernel_metadata() {
    let exporter = NotebookExporter::with_kernel(KernelSpec::evcxr());
    let ipynb = exporter.to_ipynb();

    assert!(ipynb.contains("\"language\": \"rust\""));
    assert!(ipynb.contains("\"name\": \"rust\""));
    assert!(ipynb.contains("\"display_name\": \"Rust\""));
}

#[test]
fn test_from_literate_python() {
    let content = r"
# My Analysis

Here's some code:

```python
import numpy as np
x = np.array([1, 2, 3])
```

More explanation here.
";

    let doc = LiterateDocument::parse_markdown(content);
    let exporter = NotebookExporter::from_literate(&doc);

    assert!(exporter.cell_count() >= 2);
    assert_eq!(exporter.kernel.language, "python");
}

#[test]
fn test_from_literate_rust() {
    let content = r#"
# Rust Example

```rust
fn main() {
    println!("Hello!");
}
```
"#;

    let doc = LiterateDocument::parse_markdown(content);
    let exporter = NotebookExporter::from_literate(&doc);

    assert_eq!(exporter.kernel.language, "rust");
    assert_eq!(exporter.kernel.name, "rust");
}

#[test]
fn test_cell_with_output() {
    let cell = Cell::code("print('hello')")
        .with_output(CellOutput::stream("stdout", "hello\n"))
        .with_execution_count(1);

    assert_eq!(cell.outputs.len(), 1);
    assert_eq!(cell.execution_count, Some(1));
}

#[test]
fn test_cell_source_text() {
    let cell = Cell::code("line1\nline2\nline3");
    let text = cell.source_text();

    assert!(text.contains("line1"));
    assert!(text.contains("line2"));
    assert!(text.contains("line3"));
}

#[test]
fn test_cell_type_display() {
    assert_eq!(format!("{}", CellType::Code), "code");
    assert_eq!(format!("{}", CellType::Markdown), "markdown");
    assert_eq!(format!("{}", CellType::Raw), "raw");
}

#[test]
fn test_kernel_specs() {
    let python = KernelSpec::python3();
    assert_eq!(python.language, "python");

    let rust = KernelSpec::evcxr();
    assert_eq!(rust.language, "rust");

    let julia = KernelSpec::julia();
    assert_eq!(julia.language, "julia");
}

#[test]
fn test_raw_cell() {
    let cell = Cell::raw("raw content");
    assert_eq!(cell.cell_type, CellType::Raw);
}

#[test]
fn test_notebook_json_structure() {
    let mut exporter = NotebookExporter::new();
    exporter.add_code("x = 1");

    let ipynb = exporter.to_ipynb();
    let parsed: serde_json::Value =
        serde_json::from_str(&ipynb).expect("JSON deserialization should succeed");

    assert_eq!(parsed["nbformat"], 4);
    assert_eq!(parsed["nbformat_minor"], 5);
    assert!(parsed["metadata"]["kernelspec"].is_object());
    assert!(parsed["cells"].is_array());
}

#[test]
fn test_empty_notebook() {
    let exporter = NotebookExporter::new();
    let ipynb = exporter.to_ipynb();

    let parsed: serde_json::Value =
        serde_json::from_str(&ipynb).expect("JSON deserialization should succeed");
    assert!(parsed["cells"].as_array().expect("parsing should succeed").is_empty());
}
