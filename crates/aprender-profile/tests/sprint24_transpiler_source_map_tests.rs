// Sprint 24: Transpiler Source Map Parsing (Phase 1)
//
// EXTREME TDD: RED → GREEN → REFACTOR
//
// Goal: Implement basic source map parsing for transpiled code (Python→Rust)
// - Parse JSON source map files
// - Add --transpiler-map CLI flag
// - Map Rust line numbers → Python line numbers
// - Map Rust function names → Python function names

#![allow(deprecated)] // Command::cargo_bin is deprecated but still functional

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use tempfile::TempDir;

// ============================================================================
// Test 1: Verify --transpiler-map flag is accepted
// ============================================================================

#[test]
fn test_transpiler_map_flag_accepted() {
    let temp_dir = TempDir::new().expect("test");
    let source_map = temp_dir.path().join("test.rs.sourcemap.json");

    // Create minimal valid source map
    let map_content = r#"{
        "version": 1,
        "source_language": "python",
        "source_file": "test.py",
        "generated_file": "test.rs",
        "mappings": [],
        "function_map": {}
    }"#;
    fs::write(&source_map, map_content).expect("test");

    // Test that --transpiler-map flag is accepted
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("--transpiler-map").arg(&source_map).arg("--").arg("echo").arg("test");

    // Should not error on flag parsing
    cmd.assert().success();
}

// ============================================================================
// Test 2: Parse valid source map JSON
// ============================================================================

#[test]
fn test_transpiler_map_basic_parsing() {
    let temp_dir = TempDir::new().expect("test");
    let source_map = temp_dir.path().join("simulation.rs.sourcemap.json");

    // Create realistic source map from Python→Rust transpilation
    let map_content = r#"{
        "version": 1,
        "source_language": "python",
        "source_file": "simulation.py",
        "generated_file": "simulation.rs",
        "mappings": [
            {
                "rust_line": 15,
                "rust_function": "main",
                "python_line": 10,
                "python_function": "main",
                "python_context": "def main():"
            },
            {
                "rust_line": 42,
                "rust_function": "calculate_distance",
                "python_line": 25,
                "python_function": "calculate_distance",
                "python_context": "def calculate_distance(x: int, y: int) -> float:"
            },
            {
                "rust_line": 192,
                "rust_function": "process_data",
                "python_line": 143,
                "python_function": "process_data",
                "python_context": "x: int = position[0]"
            }
        ],
        "function_map": {
            "_cse_temp_0": "temporary for: len(data) > 0",
            "_cse_temp_7_handler": "inline lambda from process_queue:89"
        }
    }"#;
    fs::write(&source_map, map_content).expect("test");

    // Run with source map
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("--transpiler-map").arg(&source_map).arg("--").arg("echo").arg("test");

    cmd.assert().success();
}

// ============================================================================
// Test 3: Handle invalid JSON gracefully
// ============================================================================

#[test]
fn test_transpiler_map_invalid_json() {
    let temp_dir = TempDir::new().expect("test");
    let source_map = temp_dir.path().join("invalid.sourcemap.json");

    // Create invalid JSON
    fs::write(&source_map, "{ this is not valid json }").expect("test");

    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("--transpiler-map").arg(&source_map).arg("--").arg("echo").arg("test");

    cmd.assert().failure().stderr(predicate::str::contains("Invalid source map JSON"));
}

// ============================================================================
// Test 4: Handle missing source map file
// ============================================================================

#[test]
fn test_transpiler_map_missing_file() {
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("--transpiler-map")
        .arg("/nonexistent/path/to/map.json")
        .arg("--")
        .arg("echo")
        .arg("test");

    cmd.assert().failure().stderr(predicate::str::contains("Source map file not found"));
}

// ============================================================================
// Test 5: Function name lookup
// ============================================================================

#[test]
fn test_transpiler_map_function_name_lookup() {
    let temp_dir = TempDir::new().expect("test");
    let source_map = temp_dir.path().join("test.sourcemap.json");

    let map_content = r#"{
        "version": 1,
        "source_language": "python",
        "source_file": "test.py",
        "generated_file": "test.rs",
        "mappings": [],
        "function_map": {
            "_cse_temp_0": "temporary for: len(data) > 0",
            "_cse_temp_7_handler": "inline lambda from process_queue:89",
            "calculate_walk_distance": "calculate_walk_distance"
        }
    }"#;
    fs::write(&source_map, map_content).expect("test");

    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("--transpiler-map").arg(&source_map).arg("--").arg("echo").arg("test");

    cmd.assert().success();
}

// ============================================================================
// Test 6: Line number lookup
// ============================================================================

#[test]
fn test_transpiler_map_line_number_lookup() {
    let temp_dir = TempDir::new().expect("test");
    let source_map = temp_dir.path().join("test.sourcemap.json");

    let map_content = r#"{
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
                "python_context": "x: int = position[0]"
            },
            {
                "rust_line": 847,
                "rust_function": "_cse_temp_7_handler",
                "python_line": 89,
                "python_function": "process_queue",
                "python_context": "queue[index].pop()"
            }
        ],
        "function_map": {}
    }"#;
    fs::write(&source_map, map_content).expect("test");

    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("--transpiler-map").arg(&source_map).arg("--").arg("echo").arg("test");

    cmd.assert().success();
}

// ============================================================================
// Test 7: Integration with basic tracing
// ============================================================================

#[test]
fn test_transpiler_map_with_tracing() {
    let temp_dir = TempDir::new().expect("test");
    let source_map = temp_dir.path().join("test.sourcemap.json");

    let map_content = r#"{
        "version": 1,
        "source_language": "python",
        "source_file": "hello.py",
        "generated_file": "hello.rs",
        "mappings": [
            {
                "rust_line": 5,
                "rust_function": "main",
                "python_line": 1,
                "python_function": "main",
                "python_context": "print('Hello from Python')"
            }
        ],
        "function_map": {
            "main": "main"
        }
    }"#;
    fs::write(&source_map, map_content).expect("test");

    // Test that tracing still works with source map loaded
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("--transpiler-map").arg(&source_map).arg("--").arg("echo").arg("hello");

    cmd.assert().success().stdout(predicate::str::contains("write("));
}

// ============================================================================
// Test 8: Handle empty mappings
// ============================================================================

#[test]
fn test_transpiler_map_empty_mappings() {
    let temp_dir = TempDir::new().expect("test");
    let source_map = temp_dir.path().join("empty.sourcemap.json");

    let map_content = r#"{
        "version": 1,
        "source_language": "python",
        "source_file": "empty.py",
        "generated_file": "empty.rs",
        "mappings": [],
        "function_map": {}
    }"#;
    fs::write(&source_map, map_content).expect("test");

    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("--transpiler-map").arg(&source_map).arg("--").arg("true");

    cmd.assert().success();
}

// ============================================================================
// Test 9: Test Python source language
// ============================================================================

#[test]
fn test_transpiler_map_python_source() {
    let temp_dir = TempDir::new().expect("test");
    let source_map = temp_dir.path().join("python.sourcemap.json");

    let map_content = r#"{
        "version": 1,
        "source_language": "python",
        "source_file": "app.py",
        "generated_file": "app.rs",
        "mappings": [
            {
                "rust_line": 100,
                "rust_function": "process_list",
                "python_line": 50,
                "python_function": "process_list",
                "python_context": "for item in items:"
            }
        ],
        "function_map": {
            "process_list": "process_list"
        }
    }"#;
    fs::write(&source_map, map_content).expect("test");

    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("--transpiler-map").arg(&source_map).arg("--").arg("echo").arg("test");

    cmd.assert().success();
}

// ============================================================================
// Test 10: Missing required fields
// ============================================================================

#[test]
fn test_transpiler_map_missing_required_fields() {
    let temp_dir = TempDir::new().expect("test");
    let source_map = temp_dir.path().join("incomplete.sourcemap.json");

    // Missing "mappings" field
    let map_content = r#"{
        "version": 1,
        "source_language": "python",
        "source_file": "test.py"
    }"#;
    fs::write(&source_map, map_content).expect("test");

    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("--transpiler-map").arg(&source_map).arg("--").arg("echo").arg("test");

    cmd.assert().failure().stderr(predicate::str::contains("Invalid source map"));
}

// ============================================================================
// Test 11: Unsupported version
// ============================================================================

#[test]
fn test_transpiler_map_unsupported_version() {
    let temp_dir = TempDir::new().expect("test");
    let source_map = temp_dir.path().join("unsupported.sourcemap.json");

    let map_content = r#"{
        "version": 999,
        "source_language": "python",
        "source_file": "test.py",
        "generated_file": "test.rs",
        "mappings": [],
        "function_map": {}
    }"#;
    fs::write(&source_map, map_content).expect("test");

    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("--transpiler-map").arg(&source_map).arg("--").arg("echo").arg("test");

    cmd.assert().failure().stderr(predicate::str::contains("Unsupported source map version"));
}
