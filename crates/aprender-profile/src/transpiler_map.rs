// Transpiler Source Map Support (Sprint 24, extended Sprint 28)
//
// Parse and manage source maps for transpiled code:
// - Python→Rust (Depyler)
// - C→Rust (Decy)
// - TypeScript→Rust
// - Any other source language
//
// Enables mapping Rust line numbers/functions back to original source language

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Source map version (currently only v1 supported)
const SUPPORTED_VERSION: u32 = 1;

/// Complete transpiler source map
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranspilerMap {
    /// Source map format version
    pub version: u32,

    /// Source language (e.g., "python", "typescript")
    pub source_language: String,

    /// Original source file path
    pub source_file: String,

    /// Generated Rust file path
    pub generated_file: String,

    /// Line mappings: Rust line → Original source location
    pub mappings: Vec<SourceMapping>,

    /// Function name mappings: Rust function → Original function/description
    pub function_map: HashMap<String, String>,
}

/// Single source mapping entry (Rust line → Python line)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceMapping {
    /// Line number in generated Rust code
    pub rust_line: usize,

    /// Function name in generated Rust code
    pub rust_function: String,

    /// Line number in original Python/TypeScript source
    pub python_line: usize,

    /// Function name in original source
    pub python_function: String,

    /// Python source code context (for debugging)
    pub python_context: String,
}

impl TranspilerMap {
    /// Load and parse source map from JSON file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_ref = path.as_ref();

        // Check file exists
        if !path_ref.exists() {
            bail!("Source map file not found: {}", path_ref.display());
        }

        // Read file
        let contents = fs::read_to_string(path_ref).context("Failed to read source map file")?;

        // Parse JSON
        let map: TranspilerMap =
            serde_json::from_str(&contents).context("Invalid source map JSON")?;

        // Validate version
        if map.version != SUPPORTED_VERSION {
            bail!(
                "Unsupported source map version: {} (expected {})",
                map.version,
                SUPPORTED_VERSION
            );
        }

        // Validate required fields
        if map.source_language.is_empty() {
            bail!("Invalid source map: missing source_language");
        }
        if map.source_file.is_empty() {
            bail!("Invalid source map: missing source_file");
        }

        Ok(map)
    }

    /// Look up original source location for a Rust line number
    pub fn lookup_line(&self, rust_line: usize) -> Option<&SourceMapping> {
        self.mappings.iter().find(|m| m.rust_line == rust_line)
    }

    /// Look up original function name for a Rust function name
    pub fn lookup_function(&self, rust_fn: &str) -> Option<&str> {
        self.function_map.get(rust_fn).map(String::as_str)
    }

    /// Get the source language (e.g., "python", "typescript")
    pub fn source_language(&self) -> &str {
        &self.source_language
    }

    /// Get the original source file path
    pub fn source_file(&self) -> &Path {
        Path::new(&self.source_file)
    }

    /// Get the generated Rust file path
    pub fn generated_file(&self) -> &Path {
        Path::new(&self.generated_file)
    }

    /// Get total number of mappings
    pub fn mapping_count(&self) -> usize {
        self.mappings.len()
    }

    /// Get total number of function mappings
    pub fn function_mapping_count(&self) -> usize {
        self.function_map.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_temp_source_map(content: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("test");
        file.write_all(content.as_bytes()).expect("test");
        file.flush().expect("test");
        file
    }

    #[test]
    fn test_parse_valid_source_map() {
        let map_json = r#"{
            "version": 1,
            "source_language": "python",
            "source_file": "test.py",
            "generated_file": "test.rs",
            "mappings": [
                {
                    "rust_line": 10,
                    "rust_function": "main",
                    "python_line": 5,
                    "python_function": "main",
                    "python_context": "def main():"
                }
            ],
            "function_map": {
                "main": "main"
            }
        }"#;

        let temp_file = create_temp_source_map(map_json);
        let map = TranspilerMap::from_file(temp_file.path()).expect("test");

        assert_eq!(map.version, 1);
        assert_eq!(map.source_language, "python");
        assert_eq!(map.source_file, "test.py");
        assert_eq!(map.generated_file, "test.rs");
        assert_eq!(map.mappings.len(), 1);
        assert_eq!(map.function_map.len(), 1);
    }

    #[test]
    fn test_lookup_line() {
        let map_json = r#"{
            "version": 1,
            "source_language": "python",
            "source_file": "test.py",
            "generated_file": "test.rs",
            "mappings": [
                {
                    "rust_line": 192,
                    "rust_function": "process_data",
                    "python_line": 143,
                    "python_function": "process_data",
                    "python_context": "x = position[0]"
                }
            ],
            "function_map": {}
        }"#;

        let temp_file = create_temp_source_map(map_json);
        let map = TranspilerMap::from_file(temp_file.path()).expect("test");

        let mapping = map.lookup_line(192).expect("test");
        assert_eq!(mapping.python_line, 143);
        assert_eq!(mapping.python_function, "process_data");

        assert!(map.lookup_line(999).is_none());
    }

    #[test]
    fn test_lookup_function() {
        let map_json = r#"{
            "version": 1,
            "source_language": "python",
            "source_file": "test.py",
            "generated_file": "test.rs",
            "mappings": [],
            "function_map": {
                "_cse_temp_0": "temporary for: len(data) > 0",
                "calculate_distance": "calculate_distance"
            }
        }"#;

        let temp_file = create_temp_source_map(map_json);
        let map = TranspilerMap::from_file(temp_file.path()).expect("test");

        assert_eq!(
            map.lookup_function("_cse_temp_0").expect("test"),
            "temporary for: len(data) > 0"
        );
        assert_eq!(map.lookup_function("calculate_distance").expect("test"), "calculate_distance");
        assert!(map.lookup_function("nonexistent").is_none());
    }

    #[test]
    fn test_invalid_json() {
        let invalid_json = "{ this is not valid json }";
        let temp_file = create_temp_source_map(invalid_json);

        let result = TranspilerMap::from_file(temp_file.path());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid source map JSON"));
    }

    #[test]
    fn test_missing_file() {
        let result = TranspilerMap::from_file("/nonexistent/path.json");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Source map file not found"));
    }

    #[test]
    fn test_unsupported_version() {
        let map_json = r#"{
            "version": 999,
            "source_language": "python",
            "source_file": "test.py",
            "generated_file": "test.rs",
            "mappings": [],
            "function_map": {}
        }"#;

        let temp_file = create_temp_source_map(map_json);
        let result = TranspilerMap::from_file(temp_file.path());

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unsupported source map version"));
    }

    #[test]
    fn test_empty_mappings() {
        let map_json = r#"{
            "version": 1,
            "source_language": "python",
            "source_file": "empty.py",
            "generated_file": "empty.rs",
            "mappings": [],
            "function_map": {}
        }"#;

        let temp_file = create_temp_source_map(map_json);
        let map = TranspilerMap::from_file(temp_file.path()).expect("test");

        assert_eq!(map.mapping_count(), 0);
        assert_eq!(map.function_mapping_count(), 0);
    }

    #[test]
    fn test_getters() {
        let map_json = r#"{
            "version": 1,
            "source_language": "typescript",
            "source_file": "app.ts",
            "generated_file": "app.rs",
            "mappings": [
                {
                    "rust_line": 10,
                    "rust_function": "main",
                    "python_line": 5,
                    "python_function": "main",
                    "python_context": "function main()"
                }
            ],
            "function_map": {
                "main": "main"
            }
        }"#;

        let temp_file = create_temp_source_map(map_json);
        let map = TranspilerMap::from_file(temp_file.path()).expect("test");

        assert_eq!(map.source_language(), "typescript");
        assert_eq!(map.source_file(), Path::new("app.ts"));
        assert_eq!(map.generated_file(), Path::new("app.rs"));
        assert_eq!(map.mapping_count(), 1);
        assert_eq!(map.function_mapping_count(), 1);
    }

    #[test]
    fn test_c_source_language_decy() {
        // Test Decy (C→Rust) transpiler source maps
        let map_json = r#"{
            "version": 1,
            "source_language": "c",
            "source_file": "algorithm.c",
            "generated_file": "algorithm.rs",
            "mappings": [
                {
                    "rust_line": 45,
                    "rust_function": "sort_array",
                    "python_line": 23,
                    "python_function": "sort_array",
                    "python_context": "for (int i = 0; i < n; i++)"
                }
            ],
            "function_map": {
                "sort_array": "sort_array",
                "_decy_temp_0": "temporary: sizeof(struct data)"
            }
        }"#;

        let temp_file = create_temp_source_map(map_json);
        let map = TranspilerMap::from_file(temp_file.path()).expect("test");

        assert_eq!(map.source_language(), "c");
        assert_eq!(map.source_file(), Path::new("algorithm.c"));
        assert_eq!(map.generated_file(), Path::new("algorithm.rs"));
        assert_eq!(map.mapping_count(), 1);
        assert_eq!(map.function_mapping_count(), 2);

        // Verify mapping lookup works for C source
        let mapping = map.lookup_line(45).expect("test");
        assert_eq!(mapping.python_line, 23);
        assert_eq!(mapping.python_function, "sort_array");

        // Verify function lookup for Decy temp variables
        assert_eq!(
            map.lookup_function("_decy_temp_0").expect("test"),
            "temporary: sizeof(struct data)"
        );
    }
}
