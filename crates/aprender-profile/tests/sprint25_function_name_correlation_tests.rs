// Sprint 25: Function Name Correlation (Phase 2)
//
// EXTREME TDD: RED → GREEN → REFACTOR
//
// Goal: Map Rust function names → Original Python/TypeScript function names
// - Integrate transpiler source maps with --function-time profiling
// - Show Python function names instead of generated Rust names in flamegraphs
// - Add --show-transpiler-context flag for verbose mapping info

#![allow(deprecated)] // Command::cargo_bin is deprecated but still functional

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use tempfile::TempDir;

// ============================================================================
// Test 1: Basic function name correlation with --function-time
// ============================================================================

#[test]
fn test_function_name_correlation_basic() {
    let temp_dir = TempDir::new().expect("test");
    let source_map = temp_dir.path().join("test.sourcemap.json");

    // Create source map with function mappings
    let map_content = r#"{
        "version": 1,
        "source_language": "python",
        "source_file": "calculator.py",
        "generated_file": "calculator.rs",
        "mappings": [],
        "function_map": {
            "calculate_distance": "calculate_distance",
            "_cse_temp_7_handler": "process_queue lambda (line 89)",
            "main": "main"
        }
    }"#;
    fs::write(&source_map, map_content).expect("test");

    // Run with --function-time and --transpiler-map
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("--transpiler-map")
        .arg(&source_map)
        .arg("--function-time")
        .arg("--")
        .arg("echo")
        .arg("test");

    // Should show Python function names in output
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("calculate_distance"))
        .stdout(predicate::str::contains("process_queue lambda"));
}

// ============================================================================
// Test 2: Function name mapping with Python source context
// ============================================================================

#[test]
fn test_function_name_with_source_context() {
    let temp_dir = TempDir::new().expect("test");
    let source_map = temp_dir.path().join("simulation.sourcemap.json");

    let map_content = r#"{
        "version": 1,
        "source_language": "python",
        "source_file": "simulation.py",
        "generated_file": "simulation.rs",
        "mappings": [
            {
                "rust_line": 192,
                "rust_function": "process_data",
                "python_line": 143,
                "python_function": "process_data",
                "python_context": "x = position[0]"
            }
        ],
        "function_map": {
            "process_data": "process_data (simulation.py:143)",
            "_run_simulation": "run_simulation",
            "_cse_temp_0": "temp: len(data) > 0"
        }
    }"#;
    fs::write(&source_map, map_content).expect("test");

    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("--transpiler-map").arg(&source_map).arg("--function-time").arg("--").arg("true");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("process_data"))
        .stdout(predicate::str::contains("run_simulation"));
}

// ============================================================================
// Test 3: --show-transpiler-context flag for verbose output
// ============================================================================

#[test]
fn test_show_transpiler_context_flag() {
    let temp_dir = TempDir::new().expect("test");
    let source_map = temp_dir.path().join("app.sourcemap.json");

    let map_content = r#"{
        "version": 1,
        "source_language": "python",
        "source_file": "app.py",
        "generated_file": "app.rs",
        "mappings": [],
        "function_map": {
            "calculate_walk_distance": "calculate_walk_distance",
            "_cse_temp_7_handler": "inline lambda from process_queue:89"
        }
    }"#;
    fs::write(&source_map, map_content).expect("test");

    // Test that --show-transpiler-context is accepted
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("--transpiler-map")
        .arg(&source_map)
        .arg("--function-time")
        .arg("--show-transpiler-context")
        .arg("--")
        .arg("echo")
        .arg("test");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("python"))
        .stdout(predicate::str::contains("Rust"));
}

// ============================================================================
// Test 4: Fallback to Rust names when no mapping exists
// ============================================================================

#[test]
fn test_fallback_to_rust_names() {
    let temp_dir = TempDir::new().expect("test");
    let source_map = temp_dir.path().join("partial.sourcemap.json");

    // Source map with only SOME functions mapped
    let map_content = r#"{
        "version": 1,
        "source_language": "python",
        "source_file": "test.py",
        "generated_file": "test.rs",
        "mappings": [],
        "function_map": {
            "main": "main"
        }
    }"#;
    fs::write(&source_map, map_content).expect("test");

    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("--transpiler-map")
        .arg(&source_map)
        .arg("--function-time")
        .arg("--")
        .arg("echo")
        .arg("test");

    // Should succeed and show mapped function "main"
    // Unmapped functions should still appear (with Rust names)
    cmd.assert().success();
}

// ============================================================================
// Test 5: Integration with --source flag
// ============================================================================

#[test]
fn test_function_correlation_with_source_flag() {
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
                "python_context": "print('Hello')"
            }
        ],
        "function_map": {
            "main": "main (hello.py:1)"
        }
    }"#;
    fs::write(&source_map, map_content).expect("test");

    // Combine --transpiler-map + --function-time + --source
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("--transpiler-map")
        .arg(&source_map)
        .arg("--function-time")
        .arg("--source")
        .arg("--")
        .arg("echo")
        .arg("hello");

    cmd.assert().success();
}

// ============================================================================
// Test 6: TypeScript source language support
// ============================================================================

#[test]
fn test_typescript_source_language() {
    let temp_dir = TempDir::new().expect("test");
    let source_map = temp_dir.path().join("app.sourcemap.json");

    let map_content = r#"{
        "version": 1,
        "source_language": "typescript",
        "source_file": "app.ts",
        "generated_file": "app.rs",
        "mappings": [],
        "function_map": {
            "processData": "processData",
            "handleRequest": "handleRequest (app.ts:42)"
        }
    }"#;
    fs::write(&source_map, map_content).expect("test");

    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("--transpiler-map")
        .arg(&source_map)
        .arg("--function-time")
        .arg("--")
        .arg("echo")
        .arg("test");

    cmd.assert().success().stdout(predicate::str::contains("processData"));
}

// ============================================================================
// Test 7: Multiple temp variables mapping
// ============================================================================

#[test]
fn test_multiple_temp_variables() {
    let temp_dir = TempDir::new().expect("test");
    let source_map = temp_dir.path().join("complex.sourcemap.json");

    let map_content = r#"{
        "version": 1,
        "source_language": "python",
        "source_file": "complex.py",
        "generated_file": "complex.rs",
        "mappings": [],
        "function_map": {
            "_cse_temp_0": "temporary: len(data) > 0",
            "_cse_temp_1": "temporary: x + y * 2",
            "_cse_temp_2": "temporary: result.is_some()",
            "_cse_temp_7_handler": "inline lambda from process_queue:89",
            "calculate_distance": "calculate_distance"
        }
    }"#;
    fs::write(&source_map, map_content).expect("test");

    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("--transpiler-map").arg(&source_map).arg("--function-time").arg("--").arg("true");

    cmd.assert().success();
}

// ============================================================================
// Test 8: Integration with statistics mode (-c)
// ============================================================================

#[test]
fn test_function_correlation_with_statistics() {
    let temp_dir = TempDir::new().expect("test");
    let source_map = temp_dir.path().join("test.sourcemap.json");

    let map_content = r#"{
        "version": 1,
        "source_language": "python",
        "source_file": "test.py",
        "generated_file": "test.rs",
        "mappings": [],
        "function_map": {
            "process_list": "process_list",
            "main": "main"
        }
    }"#;
    fs::write(&source_map, map_content).expect("test");

    // Combine --transpiler-map + --function-time + -c
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("--transpiler-map")
        .arg(&source_map)
        .arg("--function-time")
        .arg("-c")
        .arg("--")
        .arg("echo")
        .arg("test");

    cmd.assert().success();
}

// ============================================================================
// Test 9: Empty function map
// ============================================================================

#[test]
fn test_empty_function_map() {
    let temp_dir = TempDir::new().expect("test");
    let source_map = temp_dir.path().join("empty_funcs.sourcemap.json");

    let map_content = r#"{
        "version": 1,
        "source_language": "python",
        "source_file": "test.py",
        "generated_file": "test.rs",
        "mappings": [],
        "function_map": {}
    }"#;
    fs::write(&source_map, map_content).expect("test");

    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("--transpiler-map").arg(&source_map).arg("--function-time").arg("--").arg("true");

    // Should still work, just no function name translation
    cmd.assert().success();
}

// ============================================================================
// Test 10: Function correlation without --function-time
// ============================================================================

#[test]
fn test_source_map_without_function_time() {
    let temp_dir = TempDir::new().expect("test");
    let source_map = temp_dir.path().join("test.sourcemap.json");

    let map_content = r#"{
        "version": 1,
        "source_language": "python",
        "source_file": "test.py",
        "generated_file": "test.rs",
        "mappings": [],
        "function_map": {
            "main": "main"
        }
    }"#;
    fs::write(&source_map, map_content).expect("test");

    // Source map loaded but --function-time NOT enabled
    // Should still succeed, just not use function mapping
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("--transpiler-map").arg(&source_map).arg("--").arg("echo").arg("test");

    cmd.assert().success();
}
