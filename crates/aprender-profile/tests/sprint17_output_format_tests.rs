// Sprint 17: Output Format Improvements (CSV + Enhanced JSON)
// EXTREME TDD: RED phase - Integration tests for new output formats

use predicates::prelude::*;
use std::fs;
use tempfile::TempDir;

// ============================================================================
// CSV Output Format Tests
// ============================================================================

#[test]
fn test_csv_basic_output() {
    // Test basic CSV output format with --format csv
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--format").arg("csv").arg("--").arg("echo").arg("test");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("syscall,arguments,result"))
        .stdout(predicate::str::contains("write,"));
}

#[test]
fn test_csv_with_timing() {
    // Test CSV output with -T flag includes duration column
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--format").arg("csv").arg("-T").arg("--").arg("echo").arg("test");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("syscall,arguments,result,duration"))
        .stdout(predicate::str::contains("write,"));
}

#[test]
fn test_csv_with_source_correlation() {
    // Test CSV output with --source flag includes source_location column
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("test_program");

    // Create a simple test program with debug symbols
    let source = r#"
fn main() {
    println!("test");
}
"#;
    let source_file = tmp_dir.path().join("test.rs");
    fs::write(&source_file, source).expect("test");

    // Compile with debug symbols
    std::process::Command::new("rustc")
        .arg("-g")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--format").arg("csv").arg("--source").arg("--").arg(&test_program);

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("syscall,arguments,result,source_location"));
}

#[test]
fn test_csv_with_all_flags() {
    // Test CSV output with all flags combined (-T + --source)
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("test_program");

    // Create a simple test program with debug symbols
    let source = r#"
fn main() {
    println!("test");
}
"#;
    let source_file = tmp_dir.path().join("test.rs");
    fs::write(&source_file, source).expect("test");

    // Compile with debug symbols
    std::process::Command::new("rustc")
        .arg("-g")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--format").arg("csv").arg("-T").arg("--source").arg("--").arg(&test_program);

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("syscall,arguments,result,duration,source_location"));
}

#[test]
fn test_csv_with_filtering() {
    // Test CSV output works with syscall filtering
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--format").arg("csv").arg("-e").arg("trace=write").arg("--").arg("echo").arg("test");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("syscall,arguments,result"))
        .stdout(predicate::str::contains("write,"))
        .stdout(predicate::str::contains("read,").not());
}

#[test]
fn test_csv_escaping() {
    // Test CSV properly escapes special characters (commas, quotes)
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--format").arg("csv").arg("--").arg("echo").arg("test,with,commas");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("syscall,arguments,result"))
        // Arguments containing commas should be quoted
        .stdout(predicate::str::contains("\""));
}

// ============================================================================
// Enhanced JSON Output Tests
// ============================================================================

#[test]
fn test_json_with_timing() {
    // Test JSON output includes duration_us field when -T is used
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--format").arg("json").arg("-T").arg("--").arg("echo").arg("test");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("\"duration_us\":"))
        .stdout(predicate::str::contains("\"name\":"));
}

#[test]
fn test_json_with_source_correlation() {
    // Test JSON output works with --source flag (source info is optional per syscall)
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("test_program");

    // Create a simple test program with debug symbols
    let source = r#"
fn main() {
    println!("test");
}
"#;
    let source_file = tmp_dir.path().join("test.rs");
    fs::write(&source_file, source).expect("test");

    // Compile with debug symbols
    std::process::Command::new("rustc")
        .arg("-g")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--format").arg("json").arg("--source").arg("--").arg(&test_program);

    // JSON should be valid and contain syscalls (source info is optional per syscall)
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("\"syscalls\":"))
        .stdout(predicate::str::contains("\"name\":"));
}

#[test]
fn test_json_array_format() {
    // Test JSON output is valid and has correct structure
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--format").arg("json").arg("--").arg("echo").arg("test");

    cmd.assert()
        .success()
        // JSON should have version, format, syscalls, and summary fields
        .stdout(predicate::str::contains("\"version\""))
        .stdout(predicate::str::contains("\"format\""))
        .stdout(predicate::str::contains("\"syscalls\""))
        .stdout(predicate::str::contains("\"summary\""));
}

#[test]
fn test_invalid_format_error() {
    // Test that invalid format returns error
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--format").arg("invalid").arg("--").arg("echo").arg("test");

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("invalid value 'invalid' for '--format <FORMAT>'"));
}

// ============================================================================
// Statistics Mode with CSV/JSON
// ============================================================================

#[test]
fn test_csv_with_statistics_mode() {
    // Test CSV output with -c flag shows statistics summary
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--format").arg("csv").arg("-c").arg("--").arg("echo").arg("test");

    cmd.assert().success().stdout(predicate::str::contains("syscall,calls,errors"));

    // Test with timing mode adds total_time column
    let mut cmd_timing = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd_timing.arg("--format").arg("csv").arg("-c").arg("-T").arg("--").arg("echo").arg("test");

    cmd_timing
        .assert()
        .success()
        .stdout(predicate::str::contains("syscall,calls,errors,total_time"));
}

#[test]
fn test_json_with_statistics_mode() {
    // Test JSON output with -c flag shows statistics in summary
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--format").arg("json").arg("-c").arg("--").arg("echo").arg("test");

    cmd.assert()
        .success()
        // Should contain JSON structure with summary
        .stdout(predicate::str::contains("\"summary\":"))
        .stdout(predicate::str::contains("\"total_syscalls\":"))
        .stdout(predicate::str::contains("\"exit_code\":"));
}
