//! Tests for JSON batch loading (non-WASM only)

use crate::config::train::load_json_batches;
use std::io::Write;
use tempfile::NamedTempFile;

#[test]
fn test_load_json_batches_structured_format() {
    let json = r#"
{
    "examples": [
        {"input": [1.0, 2.0], "target": [3.0, 4.0]},
        {"input": [5.0, 6.0], "target": [7.0, 8.0]}
    ]
}
"#;
    let mut temp_file = NamedTempFile::new().expect("temp file creation should succeed");
    temp_file.write_all(json.as_bytes()).expect("file write should succeed");

    let result = load_json_batches(temp_file.path(), 2);
    assert!(result.is_ok());
    let batches = result.expect("operation should succeed");
    assert!(!batches.is_empty());
}

#[test]
fn test_load_json_batches_array_format() {
    let json = r#"[
    {"input": [1.0, 2.0], "target": [3.0, 4.0]},
    {"input": [5.0, 6.0], "target": [7.0, 8.0]}
]"#;
    let mut temp_file = NamedTempFile::new().expect("temp file creation should succeed");
    temp_file.write_all(json.as_bytes()).expect("file write should succeed");

    let result = load_json_batches(temp_file.path(), 1);
    assert!(result.is_ok());
    let batches = result.expect("operation should succeed");
    assert_eq!(batches.len(), 2);
}

#[test]
fn test_load_json_batches_invalid_format() {
    let json = r#"{"invalid": "format"}"#;
    let mut temp_file = NamedTempFile::new().expect("temp file creation should succeed");
    temp_file.write_all(json.as_bytes()).expect("file write should succeed");

    // Should fall back to demo data
    let result = load_json_batches(temp_file.path(), 4);
    assert!(result.is_ok());
}

#[test]
fn test_load_json_batches_nonexistent_file() {
    let result = load_json_batches(std::path::Path::new("/nonexistent/file.json"), 4);
    assert!(result.is_err());
}
