//! Pure helper functions for example validation

use anyhow::{bail, Context, Result};
use regex::Regex;
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// Get the project root directory
pub(crate) fn get_project_root() -> Result<PathBuf> {
    let current = std::env::current_dir().context("Failed to get current directory")?;

    // Look for Cargo.toml in current dir or parent
    if current.join("Cargo.toml").exists() {
        return Ok(current);
    }

    if let Some(parent) = current.parent() {
        if parent.join("Cargo.toml").exists() {
            return Ok(parent.to_path_buf());
        }
    }

    bail!("Could not find project root (no Cargo.toml found)");
}

/// Collect all example .rs files
pub(crate) fn collect_examples(examples_dir: &Path) -> Result<Vec<PathBuf>> {
    if !examples_dir.exists() {
        bail!("Examples directory does not exist: {}", examples_dir.display());
    }

    let mut examples = Vec::new();

    for entry in fs::read_dir(examples_dir).context("Failed to read examples directory")? {
        let entry = entry?;
        let path = entry.path();

        if is_rust_file(&path) {
            examples.push(path);
        }
    }

    examples.sort();
    Ok(examples)
}

/// Find all markdown files in directory
pub(crate) fn find_markdown_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();

    for entry in WalkDir::new(dir).into_iter().filter_map(std::result::Result::ok) {
        let path = entry.path();
        if is_markdown_file(path) {
            files.push(path.to_path_buf());
        }
    }

    files
}

/// Check if a file has module documentation
pub(crate) fn has_module_doc(path: &Path) -> Result<bool> {
    let content =
        fs::read_to_string(path).with_context(|| format!("Failed to read {}", path.display()))?;
    Ok(contains_module_doc(&content))
}

/// Check if example has a main function
pub(crate) fn has_main_function(path: &Path) -> Result<bool> {
    let content =
        fs::read_to_string(path).with_context(|| format!("Failed to read {}", path.display()))?;
    Ok(contains_main_function(&content))
}

/// Format a list of errors into a displayable string
pub(crate) fn format_error_list(errors: &[String], prefix: &str) -> String {
    if errors.is_empty() {
        return String::new();
    }
    format!("{}:\n  {}", prefix, errors.join("\n  "))
}

/// Extract file names from paths
pub(crate) fn extract_file_names(paths: &[PathBuf]) -> Vec<String> {
    paths
        .iter()
        .filter_map(|p| p.file_name())
        .filter_map(|n| n.to_str())
        .map(std::string::ToString::to_string)
        .collect()
}

/// Extract file stems (name without extension) from paths
pub(crate) fn extract_file_stems(paths: &[PathBuf]) -> Vec<String> {
    paths
        .iter()
        .filter_map(|p| p.file_stem())
        .filter_map(|n| n.to_str())
        .map(std::string::ToString::to_string)
        .collect()
}

/// Check if a path is a Rust file
pub(crate) fn is_rust_file(path: &Path) -> bool {
    path.extension().and_then(|s| s.to_str()) == Some("rs")
}

/// Check if a path is a Markdown file
pub(crate) fn is_markdown_file(path: &Path) -> bool {
    path.extension().and_then(|s| s.to_str()) == Some("md")
}

/// Validate `snake_case` naming
pub(crate) fn validate_snake_case(name: &str) -> bool {
    let regex = Regex::new(r"^[a-z][a-z0-9_]*$").expect("invariant: regex pattern is valid");
    regex.is_match(name)
}

/// Check if content contains a main function (pure function on string content)
pub(crate) fn contains_main_function(content: &str) -> bool {
    let main_regex = Regex::new(r"fn\s+main\s*\(").expect("invariant: regex pattern is valid");
    main_regex.is_match(content)
}

/// Check if content contains module documentation (pure function on string content)
/// Checks first 10 lines, stops at first non-comment/non-whitespace line
pub(crate) fn contains_module_doc(content: &str) -> bool {
    for line in content.lines().take(10) {
        let trimmed = line.trim();
        if trimmed.starts_with("//!") {
            return true;
        }
        // Stop at first non-comment, non-whitespace line
        if !trimmed.is_empty() && !trimmed.starts_with("//") {
            break;
        }
    }
    false
}
