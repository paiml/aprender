// Sprint 28: Decy (C→Rust) Transpiler Integration (Phase 5)
//
// EXTREME TDD: RED → GREEN → REFACTOR
//
// Goal: Add support for Decy C-to-Rust transpiler source maps
// - Support source_language: "c" in transpiler maps
// - Add generic field aliases (source_line, source_function, source_context)
// - Integration with all existing transpiler features

#![allow(deprecated)] // Command::cargo_bin is deprecated but still functional

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use tempfile::TempDir;

// ============================================================================
// Test 1: C source language is accepted
// ============================================================================

#[test]
fn test_c_source_language_accepted() {
    let temp_dir = TempDir::new().expect("test");
    let source_map = temp_dir.path().join("test.sourcemap.json");

    let map_content = r#"{
        "version": 1,
        "source_language": "c",
        "source_file": "main.c",
        "generated_file": "main.rs",
        "mappings": [],
        "function_map": {
            "main": "main"
        }
    }"#;
    fs::write(&source_map, map_content).expect("test");

    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("--transpiler-map").arg(&source_map).arg("--").arg("true");

    cmd.assert().success();
}

// ============================================================================
// Test 2: C source with function-time profiling
// ============================================================================

#[test]
fn test_c_source_with_function_time() {
    let temp_dir = TempDir::new().expect("test");
    let source_map = temp_dir.path().join("calc.sourcemap.json");

    let map_content = r#"{
        "version": 1,
        "source_language": "c",
        "source_file": "calculator.c",
        "generated_file": "calculator.rs",
        "mappings": [],
        "function_map": {
            "calculate_sum": "calculate_sum",
            "_decy_temp_1": "inline expression: a + b * c"
        }
    }"#;
    fs::write(&source_map, map_content).expect("test");

    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("--transpiler-map").arg(&source_map).arg("--function-time").arg("--").arg("true");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("calculate_sum"))
        .stdout(predicate::str::contains("inline expression"));
}

// ============================================================================
// Test 3: C source with line mappings
// ============================================================================

#[test]
fn test_c_source_with_line_mappings() {
    let temp_dir = TempDir::new().expect("test");
    let source_map = temp_dir.path().join("algo.sourcemap.json");

    let map_content = r#"{
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
            "sort_array": "sort_array"
        }
    }"#;
    fs::write(&source_map, map_content).expect("test");

    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("--transpiler-map").arg(&source_map).arg("--rewrite-stacktrace").arg("--").arg("true");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("algorithm.c"))
        .stdout(predicate::str::contains("23"));
}

// ============================================================================
// Test 4: C source with show-transpiler-context
// ============================================================================

#[test]
fn test_c_source_with_context() {
    let temp_dir = TempDir::new().expect("test");
    let source_map = temp_dir.path().join("app.sourcemap.json");

    let map_content = r#"{
        "version": 1,
        "source_language": "c",
        "source_file": "app.c",
        "generated_file": "app.rs",
        "mappings": [
            {
                "rust_line": 100,
                "rust_function": "main",
                "python_line": 50,
                "python_function": "main",
                "python_context": "int *ptr = malloc(sizeof(int) * n);"
            }
        ],
        "function_map": {
            "main": "main"
        }
    }"#;
    fs::write(&source_map, map_content).expect("test");

    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("--transpiler-map")
        .arg(&source_map)
        .arg("--function-time")
        .arg("--show-transpiler-context")
        .arg("--")
        .arg("true");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("c"))
        .stdout(predicate::str::contains("Rust"));
}

// ============================================================================
// Test 5: C source with rewrite-errors
// ============================================================================

#[test]
fn test_c_source_with_rewrite_errors() {
    let temp_dir = TempDir::new().expect("test");
    let source_map = temp_dir.path().join("error.sourcemap.json");

    let map_content = r#"{
        "version": 1,
        "source_language": "c",
        "source_file": "buggy.c",
        "generated_file": "buggy.rs",
        "mappings": [
            {
                "rust_line": 200,
                "rust_function": "process_data",
                "python_line": 75,
                "python_function": "process_data",
                "python_context": "return data[index];"
            }
        ],
        "function_map": {
            "process_data": "process_data"
        }
    }"#;
    fs::write(&source_map, map_content).expect("test");

    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("--transpiler-map").arg(&source_map).arg("--rewrite-errors").arg("--").arg("true");

    cmd.assert().success();
}

// ============================================================================
// Test 6: C source with statistics mode
// ============================================================================

#[test]
fn test_c_source_with_statistics() {
    let temp_dir = TempDir::new().expect("test");
    let source_map = temp_dir.path().join("perf.sourcemap.json");

    let map_content = r#"{
        "version": 1,
        "source_language": "c",
        "source_file": "performance.c",
        "generated_file": "performance.rs",
        "mappings": [],
        "function_map": {
            "hot_loop": "hot_loop (performance.c:150)"
        }
    }"#;
    fs::write(&source_map, map_content).expect("test");

    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("--transpiler-map").arg(&source_map).arg("-c").arg("--").arg("echo").arg("test");

    cmd.assert().success();
}

// ============================================================================
// Test 7: Multiple C function mappings (Decy temp variables)
// ============================================================================

#[test]
fn test_c_decy_temp_variables() {
    let temp_dir = TempDir::new().expect("test");
    let source_map = temp_dir.path().join("complex.sourcemap.json");

    // Decy generates temp variables like _decy_temp_N
    let map_content = r#"{
        "version": 1,
        "source_language": "c",
        "source_file": "complex.c",
        "generated_file": "complex.rs",
        "mappings": [],
        "function_map": {
            "_decy_temp_0": "temporary: sizeof(struct data)",
            "_decy_temp_1": "temporary: ptr->field",
            "_decy_temp_2": "temporary: array[i * 2 + 1]",
            "process_struct": "process_struct",
            "init_array": "init_array"
        }
    }"#;
    fs::write(&source_map, map_content).expect("test");

    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("--transpiler-map").arg(&source_map).arg("--function-time").arg("--").arg("true");

    cmd.assert().success();
}

// ============================================================================
// Test 8: C source combined with all transpiler flags
// ============================================================================

#[test]
fn test_c_source_all_flags() {
    let temp_dir = TempDir::new().expect("test");
    let source_map = temp_dir.path().join("full.sourcemap.json");

    let map_content = r#"{
        "version": 1,
        "source_language": "c",
        "source_file": "full.c",
        "generated_file": "full.rs",
        "mappings": [
            {
                "rust_line": 50,
                "rust_function": "main",
                "python_line": 25,
                "python_function": "main",
                "python_context": "int result = compute(x, y);"
            }
        ],
        "function_map": {
            "main": "main",
            "compute": "compute"
        }
    }"#;
    fs::write(&source_map, map_content).expect("test");

    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("--transpiler-map")
        .arg(&source_map)
        .arg("--function-time")
        .arg("--rewrite-stacktrace")
        .arg("--rewrite-errors")
        .arg("--show-transpiler-context")
        .arg("--")
        .arg("true");

    cmd.assert().success();
}

// ============================================================================
// Test 9: Empty C source map
// ============================================================================

#[test]
fn test_c_source_empty_mappings() {
    let temp_dir = TempDir::new().expect("test");
    let source_map = temp_dir.path().join("empty.sourcemap.json");

    let map_content = r#"{
        "version": 1,
        "source_language": "c",
        "source_file": "empty.c",
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
// Test 10: C header file reference in source_file
// ============================================================================

#[test]
fn test_c_header_file_source() {
    let temp_dir = TempDir::new().expect("test");
    let source_map = temp_dir.path().join("header.sourcemap.json");

    let map_content = r#"{
        "version": 1,
        "source_language": "c",
        "source_file": "types.h",
        "generated_file": "types.rs",
        "mappings": [
            {
                "rust_line": 10,
                "rust_function": "DataStruct",
                "python_line": 5,
                "python_function": "struct Data",
                "python_context": "typedef struct { int x; int y; } Data;"
            }
        ],
        "function_map": {
            "DataStruct": "struct Data (types.h:5)"
        }
    }"#;
    fs::write(&source_map, map_content).expect("test");

    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("--transpiler-map").arg(&source_map).arg("--function-time").arg("--").arg("true");

    cmd.assert().success().stdout(predicate::str::contains("types.h"));
}
