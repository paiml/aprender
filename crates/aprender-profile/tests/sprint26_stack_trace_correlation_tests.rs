// Sprint 26: Stack Trace Correlation (Phase 3)
//
// EXTREME TDD: RED → GREEN → REFACTOR
//
// Goal: Map Rust stack traces → Original Python/TypeScript source
// - Integrate transpiler source maps with DWARF debug info
// - Rewrite stack traces to show Python source locations
// - Add --rewrite-stacktrace flag for panic/backtrace transformation

#![allow(deprecated)] // Command::cargo_bin is deprecated but still functional

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use tempfile::TempDir;

// ============================================================================
// Test 1: --rewrite-stacktrace flag is accepted
// ============================================================================

#[test]
fn test_rewrite_stacktrace_flag_accepted() {
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
    cmd.arg("--transpiler-map").arg(&source_map).arg("--rewrite-stacktrace").arg("--").arg("true");

    // Should accept the flag without error
    cmd.assert().success();
}

// ============================================================================
// Test 2: Stack trace rewriting with line mappings
// ============================================================================

#[test]
fn test_stack_trace_rewriting_with_mappings() {
    let temp_dir = TempDir::new().expect("test");
    let source_map = temp_dir.path().join("simulation.sourcemap.json");

    // Source map with detailed line mappings
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
        .arg("--rewrite-stacktrace")
        .arg("--")
        .arg("echo")
        .arg("test");

    // Should show Python source info in output
    cmd.assert().success().stdout(predicate::str::contains("simulation.py"));
}

// ============================================================================
// Test 3: Display original Python context for mapped lines
// ============================================================================

#[test]
fn test_display_python_context() {
    let temp_dir = TempDir::new().expect("test");
    let source_map = temp_dir.path().join("app.sourcemap.json");

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
        .arg("--rewrite-stacktrace")
        .arg("--show-transpiler-context")
        .arg("--")
        .arg("true");

    // Should succeed and potentially show context
    cmd.assert().success();
}

// ============================================================================
// Test 4: Temp variable name rewriting
// ============================================================================

#[test]
fn test_temp_variable_rewriting() {
    let temp_dir = TempDir::new().expect("test");
    let source_map = temp_dir.path().join("complex.sourcemap.json");

    // Test that temp variables are mapped back to original expressions
    let map_content = r#"{
        "version": 1,
        "source_language": "python",
        "source_file": "complex.py",
        "generated_file": "complex.rs",
        "mappings": [
            {
                "rust_line": 50,
                "rust_function": "_cse_temp_0",
                "python_line": 25,
                "python_function": "check_bounds",
                "python_context": "len(data) > 0"
            }
        ],
        "function_map": {
            "_cse_temp_0": "temporary: len(data) > 0",
            "_cse_temp_1": "temporary: x + y * 2",
            "_cse_temp_7_handler": "inline lambda from process_queue:89"
        }
    }"#;
    fs::write(&source_map, map_content).expect("test");

    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("--transpiler-map")
        .arg(&source_map)
        .arg("--rewrite-stacktrace")
        .arg("--")
        .arg("echo")
        .arg("test");

    // Should show temp variable mappings
    cmd.assert().success();
}

// ============================================================================
// Test 5: Integration with --function-time
// ============================================================================

#[test]
fn test_rewrite_stacktrace_with_function_time() {
    let temp_dir = TempDir::new().expect("test");
    let source_map = temp_dir.path().join("perf.sourcemap.json");

    let map_content = r#"{
        "version": 1,
        "source_language": "python",
        "source_file": "perf.py",
        "generated_file": "perf.rs",
        "mappings": [
            {
                "rust_line": 200,
                "rust_function": "hot_function",
                "python_line": 150,
                "python_function": "hot_function",
                "python_context": "for i in range(1000000):"
            }
        ],
        "function_map": {
            "hot_function": "hot_function (perf.py:150)"
        }
    }"#;
    fs::write(&source_map, map_content).expect("test");

    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("--transpiler-map")
        .arg(&source_map)
        .arg("--rewrite-stacktrace")
        .arg("--function-time")
        .arg("--")
        .arg("true");

    // Should combine both features
    cmd.assert().success().stdout(predicate::str::contains("hot_function"));
}

// ============================================================================
// Test 6: TypeScript source language support
// ============================================================================

#[test]
fn test_rewrite_stacktrace_typescript() {
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
        .arg("--rewrite-stacktrace")
        .arg("--")
        .arg("echo")
        .arg("test");

    cmd.assert().success();
}

// ============================================================================
// Test 7: Without --rewrite-stacktrace (backward compatibility)
// ============================================================================

#[test]
fn test_backward_compatibility_without_rewrite() {
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

    // Use source map WITHOUT --rewrite-stacktrace
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("--transpiler-map").arg(&source_map).arg("--").arg("true");

    // Should still succeed
    cmd.assert().success();
}

// ============================================================================
// Test 8: Empty mappings with --rewrite-stacktrace
// ============================================================================

#[test]
fn test_rewrite_stacktrace_empty_mappings() {
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
    cmd.arg("--transpiler-map").arg(&source_map).arg("--rewrite-stacktrace").arg("--").arg("true");

    // Should succeed even with no mappings
    cmd.assert().success();
}

// ============================================================================
// Test 9: Integration with statistics mode (-c)
// ============================================================================

#[test]
fn test_rewrite_stacktrace_with_statistics() {
    let temp_dir = TempDir::new().expect("test");
    let source_map = temp_dir.path().join("stats.sourcemap.json");

    let map_content = r#"{
        "version": 1,
        "source_language": "python",
        "source_file": "stats.py",
        "generated_file": "stats.rs",
        "mappings": [],
        "function_map": {
            "process_data": "process_data",
            "main": "main"
        }
    }"#;
    fs::write(&source_map, map_content).expect("test");

    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("--transpiler-map")
        .arg(&source_map)
        .arg("--rewrite-stacktrace")
        .arg("-c")
        .arg("--")
        .arg("echo")
        .arg("test");

    cmd.assert().success();
}

// ============================================================================
// Test 10: Multiple line mappings lookup
// ============================================================================

#[test]
fn test_multiple_line_mappings() {
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
            },
            {
                "rust_line": 40,
                "rust_function": "main",
                "python_line": 35,
                "python_function": "main",
                "python_context": "print(result)"
            }
        ],
        "function_map": {
            "func_a": "func_a",
            "func_b": "func_b",
            "func_c": "func_c",
            "main": "main"
        }
    }"#;
    fs::write(&source_map, map_content).expect("test");

    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("--transpiler-map")
        .arg(&source_map)
        .arg("--rewrite-stacktrace")
        .arg("--function-time")
        .arg("--")
        .arg("true");

    cmd.assert().success();
}
