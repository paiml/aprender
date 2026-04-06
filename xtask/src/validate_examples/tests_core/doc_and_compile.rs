//! `tests_core` - Part 1

use crate::validate_examples::helpers::*;
use crate::validate_examples::results::*;
use crate::validate_examples::steps::*;
use regex::Regex;
use serial_test::serial;
use std::fs;
use std::io::Write;
use tempfile::TempDir;

#[test]
fn test_has_module_doc_with_doc() {
    let temp = TempDir::new().unwrap();
    let file = temp.path().join("example.rs");
    let mut f = fs::File::create(&file).unwrap();
    writeln!(f, "//! This is module documentation").unwrap();
    writeln!(f, "fn main() {{}}").unwrap();

    assert!(has_module_doc(&file).unwrap());
}

#[test]
fn test_has_module_doc_without_doc() {
    let temp = TempDir::new().unwrap();
    let file = temp.path().join("example.rs");
    let mut f = fs::File::create(&file).unwrap();
    writeln!(f, "// Regular comment").unwrap();
    writeln!(f, "fn main() {{}}").unwrap();

    assert!(!has_module_doc(&file).unwrap());
}

#[test]
fn test_has_module_doc_empty_file() {
    let temp = TempDir::new().unwrap();
    let file = temp.path().join("example.rs");
    fs::File::create(&file).unwrap();

    assert!(!has_module_doc(&file).unwrap());
}

#[test]
fn test_has_main_function_present() {
    let temp = TempDir::new().unwrap();
    let file = temp.path().join("example.rs");
    let mut f = fs::File::create(&file).unwrap();
    writeln!(f, "fn main() {{}}").unwrap();

    assert!(has_main_function(&file).unwrap());
}

#[test]
fn test_has_main_function_absent() {
    let temp = TempDir::new().unwrap();
    let file = temp.path().join("example.rs");
    let mut f = fs::File::create(&file).unwrap();
    writeln!(f, "fn other() {{}}").unwrap();

    assert!(!has_main_function(&file).unwrap());
}

#[test]
fn test_has_main_function_with_result() {
    let temp = TempDir::new().unwrap();
    let file = temp.path().join("example.rs");
    let mut f = fs::File::create(&file).unwrap();
    writeln!(f, "fn main() -> Result<()> {{}}").unwrap();

    assert!(has_main_function(&file).unwrap());
}

#[test]
fn test_snake_case_validation_valid() {
    let regex = Regex::new(r"^[a-z][a-z0-9_]*$").unwrap();
    assert!(regex.is_match("valid_example"));
    assert!(regex.is_match("example123"));
    assert!(regex.is_match("backend_detection"));
}

#[test]
fn test_snake_case_validation_invalid() {
    let regex = Regex::new(r"^[a-z][a-z0-9_]*$").unwrap();
    assert!(!regex.is_match("InvalidExample"));
    assert!(!regex.is_match("invalid-example"));
    assert!(!regex.is_match("123invalid"));
    assert!(!regex.is_match(""));
}

#[test]
fn test_collect_examples_empty_dir() {
    let temp = TempDir::new().unwrap();
    let examples_dir = temp.path().join("examples");
    fs::create_dir(&examples_dir).unwrap();

    let examples = collect_examples(&examples_dir).unwrap();
    assert!(examples.is_empty());
}

#[test]
fn test_collect_examples_with_files() {
    let temp = TempDir::new().unwrap();
    let examples_dir = temp.path().join("examples");
    fs::create_dir(&examples_dir).unwrap();

    // Create example files
    fs::File::create(examples_dir.join("example1.rs")).unwrap();
    fs::File::create(examples_dir.join("example2.rs")).unwrap();
    fs::File::create(examples_dir.join("not_rust.txt")).unwrap();

    let examples = collect_examples(&examples_dir).unwrap();
    assert_eq!(examples.len(), 2);
    assert!(examples.iter().all(|p| p.extension().unwrap() == "rs"));
}

#[test]
fn test_collect_examples_nonexistent_dir() {
    let temp = TempDir::new().unwrap();
    let examples_dir = temp.path().join("nonexistent");

    let result = collect_examples(&examples_dir);
    assert!(result.is_err());
}

#[test]
fn test_find_markdown_files() {
    let temp = TempDir::new().unwrap();
    let book_dir = temp.path().join("book");
    fs::create_dir(&book_dir).unwrap();

    // Create nested structure
    let src_dir = book_dir.join("src");
    fs::create_dir(&src_dir).unwrap();

    fs::File::create(book_dir.join("README.md")).unwrap();
    fs::File::create(src_dir.join("chapter1.md")).unwrap();
    fs::File::create(src_dir.join("not_md.txt")).unwrap();

    let md_files = find_markdown_files(&book_dir);
    assert_eq!(md_files.len(), 2);
    assert!(md_files.iter().all(|p| p.extension().unwrap() == "md"));
}

#[test]
fn test_validation_results_all_pass() {
    let mut results = ValidationResults::new();
    results.steps.push(StepResult {
        number: 1,
        name: "Test 1".to_string(),
        success: true,
        error: None,
    });
    results.steps.push(StepResult {
        number: 2,
        name: "Test 2".to_string(),
        success: true,
        error: None,
    });

    assert!(!results.has_failures());
}

#[test]
fn test_validation_results_with_failure() {
    let mut results = ValidationResults::new();
    results.steps.push(StepResult {
        number: 1,
        name: "Test 1".to_string(),
        success: true,
        error: None,
    });
    results.steps.push(StepResult {
        number: 2,
        name: "Test 2".to_string(),
        success: false,
        error: Some("Error message".to_string()),
    });

    assert!(results.has_failures());
}

#[test]
#[serial]
fn test_get_project_root_current_dir() {
    // This test assumes we're running in the project root or a subdirectory
    let result = get_project_root();
    assert!(result.is_ok());
    let root = result.unwrap();
    assert!(root.join("Cargo.toml").exists());
}

#[test]
fn test_validation_results_print_summary_success() {
    let mut results = ValidationResults::new();
    results.steps.push(StepResult {
        number: 1,
        name: "Test 1".to_string(),
        success: true,
        error: None,
    });
    results.steps.push(StepResult {
        number: 2,
        name: "Test 2".to_string(),
        success: true,
        error: None,
    });

    // Just verify it doesn't panic
    results.print_summary();
}

#[test]
fn test_validation_results_print_summary_with_failures() {
    let mut results = ValidationResults::new();
    results.steps.push(StepResult {
        number: 1,
        name: "Test 1".to_string(),
        success: true,
        error: None,
    });
    results.steps.push(StepResult {
        number: 2,
        name: "Test 2".to_string(),
        success: false,
        error: Some("Test error".to_string()),
    });

    // Just verify it doesn't panic
    results.print_summary();
}

#[test]
fn test_validation_results_empty() {
    let results = ValidationResults::new();
    assert!(!results.has_failures());
    results.print_summary();
}

#[test]
fn test_step_check_module_docs_success() {
    let temp = TempDir::new().unwrap();
    let file1 = temp.path().join("example1.rs");
    let file2 = temp.path().join("example2.rs");

    let mut f1 = fs::File::create(&file1).unwrap();
    writeln!(f1, "//! Module doc").unwrap();
    writeln!(f1, "fn main() {{}}").unwrap();

    let mut f2 = fs::File::create(&file2).unwrap();
    writeln!(f2, "//! Another module doc").unwrap();
    writeln!(f2, "fn main() {{}}").unwrap();

    let examples = vec![file1, file2];
    let result = step_check_module_docs(&examples);
    assert!(result.is_ok());
}

#[test]
fn test_step_check_module_docs_failure() {
    let temp = TempDir::new().unwrap();
    let file1 = temp.path().join("example1.rs");
    let file2 = temp.path().join("example2.rs");

    let mut f1 = fs::File::create(&file1).unwrap();
    writeln!(f1, "//! Module doc").unwrap();
    writeln!(f1, "fn main() {{}}").unwrap();

    let mut f2 = fs::File::create(&file2).unwrap();
    writeln!(f2, "// Regular comment, no module doc").unwrap();
    writeln!(f2, "fn main() {{}}").unwrap();

    let examples = vec![file1, file2];
    let result = step_check_module_docs(&examples);
    assert!(result.is_err());
}

#[test]
fn test_step_check_naming_conventions_success() {
    let temp = TempDir::new().unwrap();
    let file1 = temp.path().join("valid_example.rs");
    let file2 = temp.path().join("another_valid_example_123.rs");
    fs::File::create(&file1).unwrap();
    fs::File::create(&file2).unwrap();

    let examples = vec![file1, file2];
    let result = step_check_naming_conventions(&examples);
    assert!(result.is_ok());
}

#[test]
fn test_step_check_naming_conventions_failure() {
    let temp = TempDir::new().unwrap();
    let file1 = temp.path().join("valid_example.rs");
    let file2 = temp.path().join("InvalidExample.rs");
    fs::File::create(&file1).unwrap();
    fs::File::create(&file2).unwrap();

    let examples = vec![file1, file2];
    let result = step_check_naming_conventions(&examples);
    assert!(result.is_err());
}

#[test]
fn test_validation_results_add_step_success() {
    let mut results = ValidationResults::new();
    results.add_step(1, "Test step", || Ok(()));

    assert_eq!(results.steps.len(), 1);
    assert!(results.steps[0].success);
    assert!(!results.has_failures());
}

#[test]
fn test_validation_results_add_step_failure() {
    let mut results = ValidationResults::new();
    results.add_step(1, "Test step", || anyhow::bail!("Test error"));

    assert_eq!(results.steps.len(), 1);
    assert!(!results.steps[0].success);
    assert!(results.has_failures());
    assert!(results.steps[0].error.is_some());
}

#[test]
fn test_validation_results_multiple_steps() {
    let mut results = ValidationResults::new();

    results.add_step(1, "Step 1", || Ok(()));
    results.add_step(2, "Step 2", || anyhow::bail!("Error"));
    results.add_step(3, "Step 3", || Ok(()));

    assert_eq!(results.steps.len(), 3);
    assert!(results.steps[0].success);
    assert!(!results.steps[1].success);
    assert!(results.steps[2].success);
    assert!(results.has_failures());
}
