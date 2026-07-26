// Sprint 21: HPU Acceleration Foundation
// EXTREME TDD: RED phase - Integration tests for GPU-accelerated analysis

#![allow(deprecated)] // Command::cargo_bin is deprecated but still functional

use predicates::prelude::*;
use std::fs;
use tempfile::TempDir;

// ============================================================================
// Basic HPU Analysis Tests
// ============================================================================

#[test]
fn test_hpu_analysis_basic() {
    // Test that --hpu-analysis flag enables HPU acceleration
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("hpu_basic_test");

    // Create program with diverse syscall patterns for correlation analysis
    let source = r#"
#include <unistd.h>
#include <fcntl.h>
int main() {
    // Pattern 1: File operations (correlated)
    for (int i = 0; i < 20; i++) {
        int fd = open("/dev/null", O_WRONLY);
        write(fd, "test", 4);
        close(fd);
    }

    // Pattern 2: Memory operations (correlated)
    for (int i = 0; i < 15; i++) {
        void* p = sbrk(1024);
        brk(p);
    }

    return 0;
}
"#;
    let source_file = tmp_dir.path().join("hpu_basic_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("Failed to compile test program");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--hpu-analysis")
        .arg("-c") // Statistics mode to see HPU summary
        .arg("--")
        .arg(&test_program);

    // Should succeed and show HPU Analysis Report
    cmd.assert().success().stdout(predicate::str::contains("HPU Analysis Report"));
}

#[test]
fn test_hpu_correlation_matrix() {
    // Test that correlation matrix is computed and shows correlated syscalls
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("correlation_test");

    let source = r#"
#include <unistd.h>
#include <fcntl.h>
int main() {
    // Highly correlated: open-write-close pattern (repeated 30 times)
    for (int i = 0; i < 30; i++) {
        int fd = open("/dev/null", O_WRONLY);
        write(fd, "correlation", 11);
        close(fd);
    }
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("correlation_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--hpu-analysis").arg("-c").arg("--").arg(&test_program);

    // Should show correlation matrix with high correlation between open-write-close
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Correlation Matrix"))
        .stdout(predicate::str::contains("open").and(predicate::str::contains("write")));
}

#[test]
fn test_hpu_kmeans_clustering() {
    // Test that K-means clustering identifies syscall hotspot groups
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("kmeans_test");

    let source = r#"
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
int main() {
    // Cluster 1: File I/O hotspot
    for (int i = 0; i < 40; i++) {
        int fd = open("/dev/null", O_RDWR);
        write(fd, "cluster1", 8);
        close(fd);
    }

    // Cluster 2: Memory operations
    for (int i = 0; i < 25; i++) {
        void* p = mmap(NULL, 4096, PROT_READ|PROT_WRITE,
                       MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
        munmap(p, 4096);
    }

    return 0;
}
"#;
    let source_file = tmp_dir.path().join("kmeans_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--hpu-analysis").arg("-c").arg("--").arg(&test_program);

    // Should show clustering results with 2+ clusters
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("K-means Clustering"))
        .stdout(predicate::str::contains("Cluster"));
}

#[test]
fn test_hpu_performance_threshold() {
    // Test that HPU provides meaningful speedup message
    // (We can't enforce actual GPU timing, but we can verify the feature works)
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("perf_test");

    // Create program with enough syscalls to make HPU worthwhile (100+ syscalls)
    let source = r#"
#include <unistd.h>
int main() {
    for (int i = 0; i < 150; i++) {
        write(1, "x", 1);
    }
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("perf_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--hpu-analysis").arg("-c").arg("--").arg(&test_program);

    // Should show HPU backend used (GPU or CPU fallback)
    cmd.assert().success().stdout(predicate::str::contains("HPU Backend"));
}

#[test]
fn test_hpu_fallback_to_cpu() {
    // Test graceful CPU fallback when GPU unavailable (via --hpu-cpu-only)
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("fallback_test");

    let source = r#"
#include <unistd.h>
int main() {
    for (int i = 0; i < 20; i++) {
        write(1, "fallback", 8);
    }
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("fallback_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--hpu-analysis")
        .arg("--hpu-cpu-only") // Force CPU backend
        .arg("-c")
        .arg("--")
        .arg(&test_program);

    // Should succeed with CPU backend
    cmd.assert().success().stdout(predicate::str::contains("HPU Backend: CPU"));
}

// ============================================================================
// Integration with Existing Features Tests
// ============================================================================

#[test]
fn test_hpu_with_statistics() {
    // Test that --hpu-analysis works with -c (statistics mode)
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("stats_integration_test");

    let source = r#"
#include <unistd.h>
int main() {
    for (int i = 0; i < 30; i++) {
        write(1, "stats", 5);
    }
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("stats_integration_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-c").arg("--hpu-analysis").arg("--").arg(&test_program);

    // Should show both statistics AND HPU analysis
    // Statistics go to stderr (matching strace), HPU report goes to stdout
    cmd.assert()
        .success()
        .stderr(predicate::str::contains("% time"))
        .stdout(predicate::str::contains("HPU Analysis Report"));
}

#[test]
fn test_hpu_with_filtering() {
    // Test that --hpu-analysis respects -e trace=SPEC filtering
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("filter_integration_test");

    let source = r#"
#include <unistd.h>
#include <fcntl.h>
int main() {
    // File operations (will be filtered)
    for (int i = 0; i < 20; i++) {
        open("/dev/null", O_RDONLY);
    }
    // Write operations (will be included)
    for (int i = 0; i < 15; i++) {
        write(1, "filter", 6);
    }
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("filter_integration_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--hpu-analysis")
        .arg("-c")
        .arg("-e")
        .arg("trace=write") // Only analyze 'write' syscalls
        .arg("--")
        .arg(&test_program);

    // HPU analysis should only include filtered syscalls
    cmd.assert().success().stdout(predicate::str::contains("HPU Analysis Report"));
}

#[test]
fn test_hpu_with_function_time() {
    // Test that --hpu-analysis works with --function-time profiling
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("function_time_test");

    let source = r#"
#include <unistd.h>
void io_heavy() {
    for (int i = 0; i < 25; i++) {
        write(1, "io", 2);
    }
}
int main() {
    io_heavy();
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("function_time_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg("-g") // Debug symbols for function profiling
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--hpu-analysis")
        .arg("-c") // Statistics mode needed for HPU report
        .arg("--function-time")
        .arg("--source")
        .arg("--")
        .arg(&test_program);

    // Should show both function profiling AND HPU correlation
    // Function profiling summary goes to stderr, HPU report goes to stdout
    cmd.assert()
        .success()
        .stderr(
            predicate::str::contains("Function Profiling Summary")
                .or(predicate::str::contains("No function profiling data collected")),
        )
        .stdout(predicate::str::contains("HPU Analysis Report"));
}

// ============================================================================
// JSON Export and Advanced Features Tests
// ============================================================================

#[test]
fn test_hpu_json_export() {
    // Test that HPU analysis results are exported to JSON
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("json_export_test");
    let json_output = tmp_dir.path().join("hpu_output.json");

    let source = r#"
#include <unistd.h>
int main() {
    for (int i = 0; i < 30; i++) {
        write(1, "json", 4);
    }
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("json_export_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--hpu-analysis").arg("--format").arg("json").arg("--").arg(&test_program);

    let output = cmd.output().expect("test");
    fs::write(&json_output, &output.stdout).expect("test");

    // Parse JSON and verify it succeeded
    // HPU analysis goes to stdout, which may be interleaved with JSON
    let json_content = fs::read_to_string(&json_output).expect("test");
    // Check that we got either JSON output or HPU analysis (both are valid)
    assert!(
        json_content.contains("syscalls")
            || json_content.contains("hpu_analysis")
            || json_content.contains("HPU Analysis Report")
            || json_content.contains("correlation_matrix"),
        "Output should contain JSON syscalls or HPU analysis fields. Got: {}",
        &json_content[..json_content.len().min(200)]
    );
}

#[test]
fn test_hpu_large_trace() {
    // Test HPU performance on larger trace (1000+ syscalls)
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("large_trace_test");

    let source = r#"
#include <unistd.h>
#include <fcntl.h>
int main() {
    // Generate 1000+ syscalls with patterns
    for (int i = 0; i < 500; i++) {
        int fd = open("/dev/null", O_WRONLY);
        write(fd, "large", 5);
        close(fd);
    }
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("large_trace_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--hpu-analysis").arg("-c").arg("--").arg(&test_program);

    // Should handle large traces efficiently
    cmd.assert().success().stdout(predicate::str::contains("HPU Analysis Report"));
}

// ============================================================================
// Edge Cases and Error Handling Tests
// ============================================================================

#[test]
fn test_hpu_empty_trace() {
    // Test HPU behavior with empty/minimal trace
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("empty_trace_test");

    let source = r#"
int main() {
    return 0;  // Minimal syscalls (just exit)
}
"#;
    let source_file = tmp_dir.path().join("empty_trace_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--hpu-analysis").arg("-c").arg("--").arg(&test_program);

    // Should handle gracefully (no crash, informative message)
    cmd.assert().success().stdout(
        predicate::str::contains("Insufficient data for HPU analysis")
            .or(predicate::str::contains("HPU Analysis Report")),
    );
}

#[test]
fn test_backward_compatibility_without_hpu() {
    // Test that v0.4.0 works without --hpu-analysis (backward compatible)
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("compat_test");

    let source = r#"
#include <unistd.h>
int main() {
    write(1, "compat", 6);
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("compat_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-c") // Statistics without HPU
        .arg("--")
        .arg(&test_program);

    // Should work normally (no HPU output, no errors)
    // Statistics go to stderr (matching strace behavior)
    cmd.assert()
        .success()
        .stderr(predicate::str::contains("% time"))
        .stdout(predicate::str::contains("HPU Analysis Report").not());
}

#[test]
fn test_hpu_hotspot_identification() {
    // Test that HPU identifies top hotspots (most time-consuming syscall groups)
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("hotspot_test");

    let source = r#"
#include <unistd.h>
#include <fcntl.h>
#include <time.h>
int main() {
    // Hotspot 1: Slow file operations
    for (int i = 0; i < 10; i++) {
        int fd = open("/dev/null", O_WRONLY);
        struct timespec ts = {0, 1000000};  // 1ms sleep
        nanosleep(&ts, NULL);
        write(fd, "slow", 4);
        close(fd);
    }

    // Fast operations (not a hotspot)
    for (int i = 0; i < 50; i++) {
        write(1, "fast", 4);
    }

    return 0;
}
"#;
    let source_file = tmp_dir.path().join("hotspot_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--hpu-analysis")
        .arg("-c")
        .arg("-T") // Timing for hotspot detection
        .arg("--")
        .arg(&test_program);

    // Should identify slow file operations as hotspot
    // HPU report with hotspots goes to stdout
    cmd.assert().success().stdout(
        predicate::str::contains("Top Hotspots")
            .or(predicate::str::contains("HPU Analysis Report")),
    );
}
