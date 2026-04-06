//! `tests_extended` - Part 2

use crate::validate_examples::helpers::*;
use crate::validate_examples::results::*;
use crate::validate_examples::steps::*;
use serial_test::serial;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use tempfile::TempDir;

#[test]
fn test_collect_examples_preserves_full_paths() {
    let temp = TempDir::new().unwrap();
    let examples_dir = temp.path().join("examples");
    fs::create_dir(&examples_dir).unwrap();

    fs::File::create(examples_dir.join("ex1.rs")).unwrap();

    let examples = collect_examples(&examples_dir).unwrap();
    assert_eq!(examples.len(), 1);
    assert!(examples[0].is_absolute() || examples[0].starts_with(&examples_dir));
}

#[test]
fn test_find_markdown_files_empty_dir() {
    let temp = TempDir::new().unwrap();
    let book_dir = temp.path().join("book");
    fs::create_dir(&book_dir).unwrap();

    let md_files = find_markdown_files(&book_dir);
    assert!(md_files.is_empty());
}

#[test]
fn test_step_check_book_references_empty_examples() {
    let temp = TempDir::new().unwrap();
    let book_dir = temp.path().join("book");
    fs::create_dir(&book_dir).unwrap();

    let examples: Vec<PathBuf> = vec![];
    let result = step_check_book_references(&examples, &book_dir);
    assert!(result.is_ok());
}

#[test]
#[serial]
fn test_step_check_runnable_with_missing_main() {
    let temp = TempDir::new().unwrap();
    let examples_dir = temp.path().join("examples");
    fs::create_dir(&examples_dir).unwrap();

    // Create file without main function
    let file = examples_dir.join("no_main.rs");
    let mut f = fs::File::create(&file).unwrap();
    writeln!(f, "// No main function here").unwrap();
    writeln!(f, "fn helper() {{}}").unwrap();

    let project_root = get_project_root().unwrap();
    let examples = vec![file];

    let result = step_check_runnable(&examples, &project_root);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("not runnable") || err.contains("missing main"));
}

#[test]
fn test_format_error_list_single_error() {
    let errors = vec!["single error".to_string()];
    let result = format_error_list(&errors, "Problems");
    assert!(result.contains("Problems"));
    assert!(result.contains("single error"));
    assert_eq!(result.matches('\n').count(), 1); // One newline for the error
}

#[test]
fn test_validation_results_all_pass_no_errors() {
    let mut results = ValidationResults::new();
    results.add_step(1, "Test 1", || Ok(()));
    results.add_step(2, "Test 2", || Ok(()));
    results.add_step(3, "Test 3", || Ok(()));

    assert_eq!(results.steps.len(), 3);
    assert!(!results.has_failures());
    assert_eq!(count_validation_errors(&results), 0);
}

#[test]
fn test_contains_module_doc_stops_exactly_at_10_lines() {
    // If module doc is at line 11, it should not be found
    let content = "\n\n\n\n\n\n\n\n\n\nfn main() {}\n//! Too late";
    assert!(!contains_module_doc(content));
}

#[test]
fn test_contains_main_function_case_sensitive() {
    let content = "fn Main() {}"; // Capital M
                                  // Should not match since Rust is case-sensitive and Main != main
    assert!(!contains_main_function(content));
}

#[test]
fn test_validate_snake_case_single_char() {
    assert!(validate_snake_case("a"));
    assert!(validate_snake_case("z"));
    assert!(!validate_snake_case("A"));
    assert!(!validate_snake_case("Z"));
}

#[test]
fn test_format_validation_summary_single_step() {
    let results = ValidationResults {
        steps: vec![StepResult {
            number: 1,
            name: "Only step".to_string(),
            success: true,
            error: None,
        }],
    };
    let summary = format_validation_summary(&results);
    assert!(summary.contains('1'));
    assert!(summary.contains("Passed: 1"));
}

#[test]
#[serial]
fn test_step_check_runnable_error_formatting() {
    let temp = TempDir::new().unwrap();
    let file = temp.path().join("bad.rs");
    let mut f = fs::File::create(&file).unwrap();
    writeln!(f, "fn other() {{}}").unwrap(); // No main

    let project_root = get_project_root().unwrap();
    let result = step_check_runnable(&vec![file.clone()], &project_root);

    assert!(result.is_err());
    let err_str = result.unwrap_err().to_string();
    assert!(err_str.contains("runnable") || err_str.contains("main"));
}

#[test]
fn test_has_module_doc_early_return_on_code() {
    let temp = TempDir::new().unwrap();
    let file = temp.path().join("example.rs");
    let mut f = fs::File::create(&file).unwrap();
    writeln!(f, "use std::io;").unwrap(); // Code line
    writeln!(f, "//! Too late").unwrap();

    assert!(!has_module_doc(&file).unwrap());
}

#[test]
fn test_step_check_module_docs_with_multiple_missing() {
    let temp = TempDir::new().unwrap();

    let file1 = temp.path().join("no_doc1.rs");
    let file2 = temp.path().join("no_doc2.rs");
    let file3 = temp.path().join("has_doc.rs");

    let mut f1 = fs::File::create(&file1).unwrap();
    writeln!(f1, "fn main() {{}}").unwrap();

    let mut f2 = fs::File::create(&file2).unwrap();
    writeln!(f2, "fn main() {{}}").unwrap();

    let mut f3 = fs::File::create(&file3).unwrap();
    writeln!(f3, "//! Good doc").unwrap();
    writeln!(f3, "fn main() {{}}").unwrap();

    let examples = vec![file1, file2, file3];
    let result = step_check_module_docs(&examples);

    assert!(result.is_err());
    let err_str = result.unwrap_err().to_string();
    assert!(err_str.contains("missing module documentation"));
}

#[test]
fn test_step_check_book_references_multiple_invalid() {
    let temp = TempDir::new().unwrap();
    let examples_dir = temp.path().join("examples");
    let book_dir = temp.path().join("book");
    fs::create_dir_all(&examples_dir).unwrap();
    fs::create_dir_all(&book_dir).unwrap();

    let example = examples_dir.join("real_example.rs");
    fs::File::create(&example).unwrap();

    let book_file = book_dir.join("chapter.md");
    let mut f = fs::File::create(&book_file).unwrap();
    writeln!(f, "See examples/fake_one.rs and examples/fake_two.rs").unwrap();

    let result = step_check_book_references(&vec![example], &book_dir);
    assert!(result.is_err());
}

#[test]
fn test_collect_examples_single_file() {
    let temp = TempDir::new().unwrap();
    let examples_dir = temp.path().join("examples");
    fs::create_dir(&examples_dir).unwrap();

    fs::File::create(examples_dir.join("only.rs")).unwrap();

    let examples = collect_examples(&examples_dir).unwrap();
    assert_eq!(examples.len(), 1);
}

#[test]
fn test_find_markdown_files_single_file() {
    let temp = TempDir::new().unwrap();
    let book_dir = temp.path().join("book");
    fs::create_dir(&book_dir).unwrap();
    fs::File::create(book_dir.join("single.md")).unwrap();

    let md_files = find_markdown_files(&book_dir);
    assert_eq!(md_files.len(), 1);
}

#[test]
fn test_extract_file_names_single_path() {
    let paths = vec![PathBuf::from("single.rs")];
    let names = extract_file_names(&paths);
    assert_eq!(names.len(), 1);
    assert_eq!(names[0], "single.rs");
}

#[test]
fn test_extract_file_stems_single_path() {
    let paths = vec![PathBuf::from("single.rs")];
    let stems = extract_file_stems(&paths);
    assert_eq!(stems.len(), 1);
    assert_eq!(stems[0], "single");
}

#[test]
fn test_validation_results_has_failures_true() {
    let mut results = ValidationResults::new();
    results.add_step(1, "Fail", || anyhow::bail!("Error"));
    assert!(results.has_failures());
}

#[test]
fn test_validation_results_has_failures_false() {
    let mut results = ValidationResults::new();
    results.add_step(1, "Pass", || Ok(()));
    assert!(!results.has_failures());
}

#[test]
fn test_contains_module_doc_with_tabs() {
    let content = "\t//! Module doc with tab\nfn main() {}";
    assert!(contains_module_doc(content));
}

#[test]
fn test_contains_main_function_with_generics() {
    let content = "fn main<T>() {}"; // Invalid Rust, but tests regex
    assert!(!contains_main_function(content)); // Doesn't match because of <
}

#[test]
fn test_format_error_list_preserves_order() {
    let errors = vec!["first".to_string(), "second".to_string(), "third".to_string()];
    let result = format_error_list(&errors, "List");
    let first_pos = result.find("first").unwrap();
    let second_pos = result.find("second").unwrap();
    let third_pos = result.find("third").unwrap();
    assert!(first_pos < second_pos);
    assert!(second_pos < third_pos);
}

#[test]
fn test_validate_snake_case_max_length() {
    // Test with a very long but valid snake_case name
    let long_name = "a".repeat(100);
    assert!(validate_snake_case(&long_name));
}

#[test]
fn test_is_rust_file_with_path() {
    assert!(is_rust_file(&PathBuf::from("/absolute/path/file.rs")));
    assert!(is_rust_file(&PathBuf::from("relative/path/file.rs")));
}

#[test]
fn test_is_markdown_file_with_path() {
    assert!(is_markdown_file(&PathBuf::from("/absolute/path/file.md")));
    assert!(is_markdown_file(&PathBuf::from("relative/path/file.md")));
}

#[test]
fn test_count_validation_errors_empty() {
    let results = ValidationResults { steps: vec![] };
    assert_eq!(count_validation_errors(&results), 0);
}

#[test]
fn test_step_check_naming_conventions_empty_examples() {
    let examples: Vec<PathBuf> = vec![];
    let result = step_check_naming_conventions(&examples);
    assert!(result.is_ok());
}
