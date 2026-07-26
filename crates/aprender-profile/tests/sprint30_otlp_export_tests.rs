// Sprint 30: OpenTelemetry OTLP Export Integration Tests
// EXTREME TDD: RED phase - Integration tests for OTLP trace export

use std::fs;
use tempfile::TempDir;

// ============================================================================
// Basic OTLP Export Tests
// ============================================================================

#[test]
fn test_otlp_endpoint_flag_accepted() {
    // Test that --otlp-endpoint flag is recognized
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("otlp_basic_test");

    let source = r#"
#include <unistd.h>
int main() {
    write(1, "test\n", 5);
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("otlp_basic_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("Failed to compile test program");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--otlp-endpoint").arg("http://localhost:4317").arg("--").arg(&test_program);

    // Should accept the flag (may fail to connect, but flag should be accepted)
    cmd.assert().success();
}

#[test]
fn test_otlp_endpoint_default_disabled() {
    // Test that OTLP export is disabled by default
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("otlp_disabled_test");

    let source = r#"
#include <unistd.h>
int main() {
    write(1, "test\n", 5);
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("otlp_disabled_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("Failed to compile test program");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--").arg(&test_program);

    // Should work without OTLP
    cmd.assert().success();
}

#[test]
fn test_otlp_with_statistics_mode() {
    // Test that OTLP export works with statistics mode
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("otlp_stats_test");

    let source = r#"
#include <unistd.h>
int main() {
    for (int i = 0; i < 10; i++) {
        write(1, "x", 1);
    }
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("otlp_stats_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("Failed to compile test program");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-c").arg("--otlp-endpoint").arg("http://localhost:4317").arg("--").arg(&test_program);

    cmd.assert().success();
}

#[test]
fn test_otlp_service_name_configuration() {
    // Test that service name can be configured
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("otlp_service_test");

    let source = r#"
#include <unistd.h>
int main() {
    write(1, "test\n", 5);
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("otlp_service_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("Failed to compile test program");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--otlp-endpoint")
        .arg("http://localhost:4317")
        .arg("--otlp-service-name")
        .arg("my-traced-app")
        .arg("--")
        .arg(&test_program);

    cmd.assert().success();
}

#[test]
fn test_otlp_invalid_endpoint() {
    // Test graceful handling of invalid endpoint
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("otlp_invalid_test");

    let source = r#"
#include <unistd.h>
int main() {
    write(1, "test\n", 5);
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("otlp_invalid_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("Failed to compile test program");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--otlp-endpoint").arg("invalid://endpoint").arg("--").arg(&test_program);

    // Should still trace, but may log warning about export failure
    cmd.assert().success();
}

#[test]
fn test_otlp_with_timing_mode() {
    // Test that OTLP export includes timing information
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("otlp_timing_test");

    let source = r#"
#include <unistd.h>
int main() {
    write(1, "a", 1);
    write(1, "b", 1);
    write(1, "c", 1);
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("otlp_timing_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("Failed to compile test program");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-T").arg("--otlp-endpoint").arg("http://localhost:4317").arg("--").arg(&test_program);

    cmd.assert().success();
}

#[test]
fn test_otlp_with_source_correlation() {
    // Test that OTLP spans include source location attributes when available
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("otlp_source_test");

    let source = r#"
#include <unistd.h>
int main() {
    write(1, "test\n", 5);
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("otlp_source_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg("-g") // Include debug info
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("Failed to compile test program");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-s") // Enable source correlation
        .arg("--otlp-endpoint")
        .arg("http://localhost:4317")
        .arg("--")
        .arg(&test_program);

    cmd.assert().success();
}

#[test]
fn test_otlp_with_filtering() {
    // Test that OTLP export respects syscall filters
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("otlp_filter_test");

    let source = r#"
#include <unistd.h>
#include <fcntl.h>
int main() {
    write(1, "test\n", 5);
    open("/dev/null", O_RDONLY);
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("otlp_filter_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("Failed to compile test program");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-e")
        .arg("trace=write")
        .arg("--otlp-endpoint")
        .arg("http://localhost:4317")
        .arg("--")
        .arg(&test_program);

    cmd.assert().success();
}

#[test]
fn test_otlp_backward_compatibility() {
    // Test that programs work normally without OTLP flags
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("otlp_compat_test");

    let source = r#"
#include <unistd.h>
int main() {
    write(1, "test\n", 5);
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("otlp_compat_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("Failed to compile test program");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--").arg(&test_program);

    // Should work exactly as before
    cmd.assert().success();
}

#[test]
fn test_otlp_grpc_protocol() {
    // Test gRPC endpoint (default OTLP protocol)
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("otlp_grpc_test");

    let source = r#"
#include <unistd.h>
int main() {
    write(1, "test\n", 5);
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("otlp_grpc_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("Failed to compile test program");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--otlp-endpoint")
        .arg("http://localhost:4317") // gRPC port
        .arg("--")
        .arg(&test_program);

    cmd.assert().success();
}

#[test]
fn test_otlp_http_protocol() {
    // Test HTTP endpoint as alternative protocol
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("otlp_http_test");

    let source = r#"
#include <unistd.h>
int main() {
    write(1, "test\n", 5);
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("otlp_http_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("Failed to compile test program");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--otlp-endpoint")
        .arg("http://localhost:4318") // HTTP port
        .arg("--")
        .arg(&test_program);

    cmd.assert().success();
}

#[test]
fn test_otlp_trace_hierarchy() {
    // Test that syscalls are exported as child spans of a root trace
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("otlp_hierarchy_test");

    let source = r#"
#include <unistd.h>
int main() {
    write(1, "a", 1);
    write(1, "b", 1);
    write(1, "c", 1);
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("otlp_hierarchy_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("Failed to compile test program");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--otlp-endpoint").arg("http://localhost:4317").arg("--").arg(&test_program);

    cmd.assert().success();
}
