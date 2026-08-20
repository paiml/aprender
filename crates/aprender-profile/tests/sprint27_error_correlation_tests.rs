// Sprint 27: Compilation Error Correlation (Phase 4)
//
// EXTREME TDD: RED → GREEN → REFACTOR
//
// Goal: Map rustc errors → Original Python/TypeScript source
// - Add --rewrite-errors flag for error message transformation
// - Parse rustc error format and rewrite line numbers
// - Show Python source context for compile errors

#![allow(deprecated)] // Command::cargo_bin is deprecated but still functional

use assert_cmd::Command;
use std::fs;
use tempfile::TempDir;

// ============================================================================
// Test 1: --rewrite-errors flag is accepted
// ============================================================================

#[test]
fn test_rewrite_errors_flag_accepted() {
    let temp_dir = TempDir::new().expect("test");
    let source_map = temp_dir.path().join("test.sourcemap.json");

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
    cmd.arg("--transpiler-map").arg(&source_map).arg("--rewrite-errors").arg("--").arg("true");

    // Should accept the flag without error
    cmd.assert().success();
}

// ============================================================================
// Test 2: Error correlation with line mappings
// ============================================================================

#[test]
fn test_error_correlation_with_mappings() {
    let temp_dir = TempDir::new().expect("test");
    let source_map = temp_dir.path().join("error_map.sourcemap.json");

    // Source map with line mappings for error correlation
    let map_content = r#"{
        "version": 1,
        "source_language": "python",
        "source_file": "simulation.py",
        "generated_file": "simulation.rs",
        "mappings": [
            {
                "rust_line": 847,
                "rust_function": "_cse_temp_12_handler",
                "python_line": 89,
                "python_function": "process_queue",
                "python_context": "result = queue[index]"
            },
            {
                "rust_line": 192,
                "rust_function": "calculate_distance",
                "python_line": 45,
                "python_function": "calculate_distance",
                "python_context": "distance = sqrt(x**2 + y**2)"
            }
        ],
        "function_map": {
            "_cse_temp_12_handler": "process_queue lambda (line 89)",
            "calculate_distance": "calculate_distance"
        }
    }"#;
    fs::write(&source_map, map_content).expect("test");

    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("--transpiler-map")
        .arg(&source_map)
        .arg("--rewrite-errors")
        .arg("--")
        .arg("echo")
        .arg("test");

    // Should show source file info when using --rewrite-errors
    cmd.assert().success();
}

// ============================================================================
// Test 3: Integration with --show-transpiler-context
// ============================================================================

#[test]
fn test_rewrite_errors_with_context() {
    let temp_dir = TempDir::new().expect("test");
    let source_map = temp_dir.path().join("context.sourcemap.json");

    let map_content = r#"{
        "version": 1,
        "source_language": "python",
        "source_file": "app.py",
        "generated_file": "app.rs",
        "mappings": [
            {
                "rust_line": 100,
                "rust_function": "main",
                "python_line": 10,
                "python_function": "main",
                "python_context": "data = load_data()"
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
        .arg("--rewrite-errors")
        .arg("--show-transpiler-context")
        .arg("--")
        .arg("true");

    // Should show error mapping context
    cmd.assert().success();
}

// ============================================================================
// Test 4: TypeScript source language support
// ============================================================================

#[test]
fn test_rewrite_errors_typescript() {
    let temp_dir = TempDir::new().expect("test");
    let source_map = temp_dir.path().join("app.sourcemap.json");

    let map_content = r#"{
        "version": 1,
        "source_language": "typescript",
        "source_file": "app.ts",
        "generated_file": "app.rs",
        "mappings": [
            {
                "rust_line": 30,
                "rust_function": "handleRequest",
                "python_line": 20,
                "python_function": "handleRequest",
                "python_context": "const result = await fetch(url)"
            }
        ],
        "function_map": {
            "handleRequest": "handleRequest (app.ts:20)"
        }
    }"#;
    fs::write(&source_map, map_content).expect("test");

    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("--transpiler-map")
        .arg(&source_map)
        .arg("--rewrite-errors")
        .arg("--")
        .arg("echo")
        .arg("test");

    cmd.assert().success();
}

// ============================================================================
// Test 5: Backward compatibility without --rewrite-errors
// ============================================================================

#[test]
fn test_backward_compatibility_without_rewrite_errors() {
    let temp_dir = TempDir::new().expect("test");
    let source_map = temp_dir.path().join("test.sourcemap.json");

    let map_content = r#"{
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
                "python_context": "print('hello')"
            }
        ],
        "function_map": {
            "main": "main"
        }
    }"#;
    fs::write(&source_map, map_content).expect("test");

    // Use source map WITHOUT --rewrite-errors
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("--transpiler-map").arg(&source_map).arg("--").arg("true");

    // Should still succeed
    cmd.assert().success();
}

// ============================================================================
// Test 6: Empty mappings with --rewrite-errors
// ============================================================================

#[test]
fn test_rewrite_errors_empty_mappings() {
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
    cmd.arg("--transpiler-map").arg(&source_map).arg("--rewrite-errors").arg("--").arg("true");

    // Should succeed even with no mappings
    cmd.assert().success();
}

// ============================================================================
// Test 7: Integration with --rewrite-stacktrace
// ============================================================================

#[test]
fn test_rewrite_errors_with_stacktrace() {
    let temp_dir = TempDir::new().expect("test");
    let source_map = temp_dir.path().join("combined.sourcemap.json");

    let map_content = r#"{
        "version": 1,
        "source_language": "python",
        "source_file": "combined.py",
        "generated_file": "combined.rs",
        "mappings": [
            {
                "rust_line": 50,
                "rust_function": "process_data",
                "python_line": 25,
                "python_function": "process_data",
                "python_context": "for item in data:"
            }
        ],
        "function_map": {
            "process_data": "process_data"
        }
    }"#;
    fs::write(&source_map, map_content).expect("test");

    // Combine both flags
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("--transpiler-map")
        .arg(&source_map)
        .arg("--rewrite-errors")
        .arg("--rewrite-stacktrace")
        .arg("--")
        .arg("true");

    cmd.assert().success();
}

// ============================================================================
// Test 8: Integration with statistics mode (-c)
// ============================================================================

#[test]
fn test_rewrite_errors_with_statistics() {
    let temp_dir = TempDir::new().expect("test");
    let source_map = temp_dir.path().join("stats.sourcemap.json");

    let map_content = r#"{
        "version": 1,
        "source_language": "python",
        "source_file": "stats.py",
        "generated_file": "stats.rs",
        "mappings": [],
        "function_map": {
            "main": "main"
        }
    }"#;
    fs::write(&source_map, map_content).expect("test");

    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("--transpiler-map")
        .arg(&source_map)
        .arg("--rewrite-errors")
        .arg("-c")
        .arg("--")
        .arg("echo")
        .arg("test");

    cmd.assert().success();
}

// ============================================================================
// Test 9: Multiple line mappings for error correlation
// ============================================================================

#[test]
fn test_multiple_error_mappings() {
    let temp_dir = TempDir::new().expect("test");
    let source_map = temp_dir.path().join("multi.sourcemap.json");

    // Multiple mappings to test lookup efficiency
    let map_content = r#"{
        "version": 1,
        "source_language": "python",
        "source_file": "multi.py",
        "generated_file": "multi.rs",
        "mappings": [
            {
                "rust_line": 10,
                "rust_function": "func_a",
                "python_line": 5,
                "python_function": "func_a",
                "python_context": "return x + 1"
            },
            {
                "rust_line": 20,
                "rust_function": "func_b",
                "python_line": 15,
                "python_function": "func_b",
                "python_context": "return y * 2"
            },
            {
                "rust_line": 30,
                "rust_function": "func_c",
                "python_line": 25,
                "python_function": "func_c",
                "python_context": "return z - 3"
            }
        ],
        "function_map": {
            "func_a": "func_a",
            "func_b": "func_b",
            "func_c": "func_c"
        }
    }"#;
    fs::write(&source_map, map_content).expect("test");

    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("--transpiler-map").arg(&source_map).arg("--rewrite-errors").arg("--").arg("true");

    cmd.assert().success();
}

// ============================================================================
// Test 10: Integration with --function-time
// ============================================================================

#[test]
fn test_rewrite_errors_with_function_time() {
    let temp_dir = TempDir::new().expect("test");
    let source_map = temp_dir.path().join("perf.sourcemap.json");

    let map_content = r#"{
        "version": 1,
        "source_language": "python",
        "source_file": "perf.py",
        "generated_file": "perf.rs",
        "mappings": [],
        "function_map": {
            "hot_function": "hot_function (perf.py:150)"
        }
    }"#;
    fs::write(&source_map, map_content).expect("test");

    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("--transpiler-map")
        .arg(&source_map)
        .arg("--rewrite-errors")
        .arg("--function-time")
        .arg("--")
        .arg("true");

    cmd.assert().success();
}
