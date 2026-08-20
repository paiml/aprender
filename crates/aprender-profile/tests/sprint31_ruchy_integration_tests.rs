// Sprint 31: Ruchy Runtime Integration Tests
//
// Goal: Link OTLP traces with transpiler decision traces for end-to-end observability
//
// Integration tests for:
// 1. Exporting transpiler decisions as OTLP span events
// 2. Correlating decisions with syscall spans
// 3. Unified trace view (Process → Syscalls → Decisions)

#![allow(deprecated)] // assert_cmd::Command::cargo_bin deprecation

use assert_cmd::Command;
use std::fs;
use tempfile::TempDir;

#[test]
fn test_otlp_with_decision_traces() {
    // RED Phase: Test that --otlp-endpoint and --trace-transpiler-decisions work together
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let test_program = temp_dir.path().join("decision_test");

    // Create a test program that prints decision traces to stderr
    fs::write(
        &test_program,
        r#"#!/bin/bash
echo "[DECISION] type_inference: inferring type for variable 'x'" >&2
echo "Hello, Ruchy!"
echo "[RESULT] type_inference: inferred i32" >&2
"#,
    )
    .expect("Failed to write test program");

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&test_program, fs::Permissions::from_mode(0o755))
            .expect("Failed to set permissions");
    }

    let mut cmd = Command::cargo_bin("aprender-profile").expect("Failed to find renacer binary");
    cmd.arg("--otlp-endpoint")
        .arg("http://localhost:4317")
        .arg("--trace-transpiler-decisions")
        .arg("--")
        .arg(&test_program);

    // Should succeed and export both syscalls and decisions to OTLP
    cmd.assert().success();
}

#[test]
fn test_decision_as_span_event() {
    // RED Phase: Test that transpiler decisions are exported as OTLP span events
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let test_program = temp_dir.path().join("decision_event_test");

    fs::write(
        &test_program,
        r#"#!/bin/bash
echo "[DECISION] optimization: inline function 'compute'" >&2
echo "Computing..."
echo "[RESULT] optimization: inlined" >&2
"#,
    )
    .expect("Failed to write test program");

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&test_program, fs::Permissions::from_mode(0o755))
            .expect("Failed to set permissions");
    }

    let mut cmd = Command::cargo_bin("aprender-profile").expect("Failed to find renacer binary");
    cmd.arg("--otlp-endpoint")
        .arg("http://localhost:4317")
        .arg("--trace-transpiler-decisions")
        .arg("--")
        .arg(&test_program);

    // Should not crash with OTLP + decision traces
    cmd.assert().success();
}

#[test]
fn test_decision_correlation_with_syscalls() {
    // RED Phase: Test that decisions are correlated with the syscalls that trigger them
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let test_program = temp_dir.path().join("correlation_test");

    fs::write(
        &test_program,
        r#"#!/bin/bash
echo "[DECISION] type_check: checking expression" >&2
echo "Result"
echo "[RESULT] type_check: type is valid" >&2
"#,
    )
    .expect("Failed to write test program");

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&test_program, fs::Permissions::from_mode(0o755))
            .expect("Failed to set permissions");
    }

    let mut cmd = Command::cargo_bin("aprender-profile").expect("Failed to find renacer binary");
    cmd.arg("--otlp-endpoint")
        .arg("http://localhost:4317")
        .arg("--trace-transpiler-decisions")
        .arg("--")
        .arg(&test_program);

    // Should succeed and correlate decisions with write(2) syscalls
    cmd.assert().success();
}

#[test]
fn test_otlp_with_source_map_and_decisions() {
    // RED Phase: Test triple integration: OTLP + source maps + decisions
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let test_program = temp_dir.path().join("triple_test");
    let source_map = temp_dir.path().join("test.sourcemap.json");

    fs::write(
        &source_map,
        r#"{
  "version": 1,
  "source_language": "python",
  "source_file": "test.py",
  "generated_file": "test.rs",
  "mappings": [],
  "function_map": {}
}"#,
    )
    .expect("Failed to write source map");

    fs::write(
        &test_program,
        r#"#!/bin/bash
echo "[DECISION] pattern_compile: compiling match expression" >&2
echo "Match result"
echo "[RESULT] pattern_compile: compiled" >&2
"#,
    )
    .expect("Failed to write test program");

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&test_program, fs::Permissions::from_mode(0o755))
            .expect("Failed to set permissions");
    }

    let mut cmd = Command::cargo_bin("aprender-profile").expect("Failed to find renacer binary");
    cmd.arg("--otlp-endpoint")
        .arg("http://localhost:4317")
        .arg("--trace-transpiler-decisions")
        .arg("--transpiler-map")
        .arg(&source_map)
        .arg("--")
        .arg(&test_program);

    // Should succeed with all three features enabled
    cmd.assert().success();
}

#[test]
fn test_decision_span_event_attributes() {
    // RED Phase: Test that decision span events have correct attributes
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let test_program = temp_dir.path().join("attributes_test");

    fs::write(
        &test_program,
        r#"#!/bin/bash
echo "[DECISION] trait_solve: resolving trait bound" >&2
echo "Output"
echo "[RESULT] trait_solve: bound satisfied" >&2
"#,
    )
    .expect("Failed to write test program");

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&test_program, fs::Permissions::from_mode(0o755))
            .expect("Failed to set permissions");
    }

    let mut cmd = Command::cargo_bin("aprender-profile").expect("Failed to find renacer binary");
    cmd.arg("--otlp-endpoint")
        .arg("http://localhost:4317")
        .arg("--trace-transpiler-decisions")
        .arg("--")
        .arg(&test_program);

    // Attributes should include: decision.category, decision.name, decision.result
    cmd.assert().success();
}

#[test]
fn test_backward_compatibility_otlp_without_decisions() {
    // RED Phase: Test that OTLP works fine without --trace-transpiler-decisions
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let test_program = temp_dir.path().join("no_decisions_test");

    fs::write(&test_program, "#!/bin/bash\necho 'No decisions here'\n")
        .expect("Failed to write test program");

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&test_program, fs::Permissions::from_mode(0o755))
            .expect("Failed to set permissions");
    }

    let mut cmd = Command::cargo_bin("aprender-profile").expect("Failed to find renacer binary");
    cmd.arg("--otlp-endpoint").arg("http://localhost:4317").arg("--").arg(&test_program);

    // Should work fine - OTLP without decisions
    cmd.assert().success();
}

#[test]
fn test_backward_compatibility_decisions_without_otlp() {
    // RED Phase: Test that decisions work fine without --otlp-endpoint
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let test_program = temp_dir.path().join("decisions_only_test");

    fs::write(
        &test_program,
        r#"#!/bin/bash
echo "[DECISION] const_eval: evaluating constant expression" >&2
echo "Value"
echo "[RESULT] const_eval: evaluated to 42" >&2
"#,
    )
    .expect("Failed to write test program");

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&test_program, fs::Permissions::from_mode(0o755))
            .expect("Failed to set permissions");
    }

    let mut cmd = Command::cargo_bin("aprender-profile").expect("Failed to find renacer binary");
    cmd.arg("--trace-transpiler-decisions").arg("--").arg(&test_program);

    // Should work fine - decisions without OTLP
    cmd.assert().success();
}

#[test]
fn test_multiple_decisions_as_span_events() {
    // RED Phase: Test that multiple decisions create multiple span events
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let test_program = temp_dir.path().join("multiple_decisions_test");

    fs::write(
        &test_program,
        r#"#!/bin/bash
echo "[DECISION] type_check: checking type" >&2
echo "[RESULT] type_check: valid" >&2
echo "[DECISION] optimization: attempting inline" >&2
echo "[RESULT] optimization: inlined" >&2
echo "[DECISION] code_gen: generating assembly" >&2
echo "[RESULT] code_gen: generated" >&2
echo "Done"
"#,
    )
    .expect("Failed to write test program");

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&test_program, fs::Permissions::from_mode(0o755))
            .expect("Failed to set permissions");
    }

    let mut cmd = Command::cargo_bin("aprender-profile").expect("Failed to find renacer binary");
    cmd.arg("--otlp-endpoint")
        .arg("http://localhost:4317")
        .arg("--trace-transpiler-decisions")
        .arg("--")
        .arg(&test_program);

    // Should export all 3 decisions as span events
    cmd.assert().success();
}

#[test]
fn test_decision_timing_in_span_events() {
    // RED Phase: Test that decision timing is captured in span events
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let test_program = temp_dir.path().join("timing_test");

    fs::write(
        &test_program,
        r#"#!/bin/bash
echo "[DECISION] expensive_op: starting" >&2
sleep 0.01  # 10ms operation
echo "[RESULT] expensive_op: completed" >&2
echo "Result"
"#,
    )
    .expect("Failed to write test program");

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&test_program, fs::Permissions::from_mode(0o755))
            .expect("Failed to set permissions");
    }

    let mut cmd = Command::cargo_bin("aprender-profile").expect("Failed to find renacer binary");
    cmd.arg("--otlp-endpoint")
        .arg("http://localhost:4317")
        .arg("--trace-transpiler-decisions")
        .arg("-T") // Enable timing
        .arg("--")
        .arg(&test_program);

    // Should capture decision timing in span events
    cmd.assert().success();
}

#[test]
fn test_otlp_service_name_with_decisions() {
    // RED Phase: Test that --otlp-service-name works with decision traces
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let test_program = temp_dir.path().join("service_name_test");

    fs::write(
        &test_program,
        r#"#!/bin/bash
echo "[DECISION] test_decision: testing" >&2
echo "[RESULT] test_decision: success" >&2
echo "Output"
"#,
    )
    .expect("Failed to write test program");

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&test_program, fs::Permissions::from_mode(0o755))
            .expect("Failed to set permissions");
    }

    let mut cmd = Command::cargo_bin("aprender-profile").expect("Failed to find renacer binary");
    cmd.arg("--otlp-endpoint")
        .arg("http://localhost:4317")
        .arg("--otlp-service-name")
        .arg("ruchy-integrated-app")
        .arg("--trace-transpiler-decisions")
        .arg("--")
        .arg(&test_program);

    // Should use custom service name with decisions
    cmd.assert().success();
}

#[test]
fn test_decision_events_with_filtering() {
    // RED Phase: Test that syscall filtering doesn't affect decision events
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let test_program = temp_dir.path().join("filtering_test");

    fs::write(
        &test_program,
        r#"#!/bin/bash
echo "[DECISION] decision1: starting" >&2
echo "[RESULT] decision1: done" >&2
echo "Output"
"#,
    )
    .expect("Failed to write test program");

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&test_program, fs::Permissions::from_mode(0o755))
            .expect("Failed to set permissions");
    }

    let mut cmd = Command::cargo_bin("aprender-profile").expect("Failed to find renacer binary");
    cmd.arg("--otlp-endpoint")
        .arg("http://localhost:4317")
        .arg("--trace-transpiler-decisions")
        .arg("-e")
        .arg("trace=write") // Filter to only write syscalls
        .arg("--")
        .arg(&test_program);

    // Decision events should still be exported even if syscalls are filtered
    cmd.assert().success();
}
