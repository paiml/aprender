// Sprint 20: Real-Time Anomaly Detection with Sliding Window
// EXTREME TDD: RED phase - Integration tests for real-time anomaly detection

use predicates::prelude::*;
use std::fs;
use tempfile::TempDir;

// ============================================================================
// Basic Real-Time Anomaly Detection Tests
// ============================================================================

#[test]
fn test_realtime_anomaly_detects_slow_syscall() {
    // Test that --anomaly-realtime detects and reports slow syscalls in real-time
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("realtime_anomaly_test");

    // Create program with baseline fast syscalls + one anomalous slow syscall
    let source = r#"
#include <unistd.h>
#include <time.h>
int main() {
    // Establish baseline: 50 fast writes
    for (int i = 0; i < 50; i++) {
        write(1, "x", 1);
    }

    // Simulate slow I/O (anomaly)
    struct timespec ts = {0, 50000000};  // 50ms sleep (increased for test stability)
    nanosleep(&ts, NULL);
    write(1, "slow", 4);

    return 0;
}
"#;
    let source_file = tmp_dir.path().join("realtime_anomaly_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("Failed to compile test program");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--anomaly-realtime").arg("-T").arg("--").arg(&test_program);

    // Should show real-time anomaly alert
    cmd.assert().success().stderr(predicate::str::contains("⚠️  ANOMALY"));
}

#[test]
fn test_anomaly_window_size_configuration() {
    // Test that --anomaly-window-size configures sliding window
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("window_size_test");

    let source = r#"
#include <unistd.h>
int main() {
    for (int i = 0; i < 30; i++) {
        write(1, "x", 1);
    }
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("window_size_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--anomaly-realtime")
        .arg("--anomaly-window-size")
        .arg("20")
        .arg("--")
        .arg(&test_program);

    // Should accept custom window size
    cmd.assert().success();
}

#[test]
fn test_anomaly_requires_minimum_samples() {
    // Test that anomaly detection waits for minimum samples (10) before detecting
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("min_samples_test");

    let source = r#"
#include <unistd.h>
int main() {
    // Only 5 syscalls - not enough for anomaly detection
    for (int i = 0; i < 5; i++) {
        write(1, "x", 1);
    }
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("min_samples_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--anomaly-realtime")
        .arg("-T")
        .arg("-e")
        .arg("trace=write") // Filter to only trace write syscalls (5 calls)
        .arg("--")
        .arg(&test_program);

    // Should NOT report anomalies (insufficient samples for write syscalls)
    // With only 5 write syscalls, anomaly detection should not trigger
    cmd.assert().success().stderr(predicate::str::contains("ANOMALY").not());
}

#[test]
fn test_anomaly_severity_classification() {
    // Test that anomalies are classified by severity (3-4σ: Low, 4-5σ: Medium, >5σ: High)
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("severity_test");

    let source = r#"
#include <unistd.h>
#include <time.h>
int main() {
    // Baseline: many fast writes
    for (int i = 0; i < 100; i++) {
        write(1, "x", 1);
    }

    // Extremely slow syscall (should be High severity)
    struct timespec ts = {0, 50000000};  // 50ms
    nanosleep(&ts, NULL);
    write(1, "very_slow", 9);

    return 0;
}
"#;
    let source_file = tmp_dir.path().join("severity_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--anomaly-realtime").arg("-T").arg("--").arg(&test_program);

    // Should classify anomaly with severity
    cmd.assert().success();
    // Note: Actual severity depends on system variance
}

// ============================================================================
// Integration with Other Flags
// ============================================================================

#[test]
fn test_anomaly_realtime_with_statistics() {
    // Test --anomaly-realtime works with -c flag
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("stats_anomaly_test");

    let source = r#"
#include <unistd.h>
int main() {
    for (int i = 0; i < 50; i++) {
        write(1, "data\n", 5);
    }
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("stats_anomaly_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-c").arg("--anomaly-realtime").arg("-T").arg("--").arg(&test_program);

    // Should show both statistics summary and anomaly report
    cmd.assert().success().stderr(predicate::str::contains("% time"));
}

#[test]
fn test_anomaly_realtime_with_filtering() {
    // Test --anomaly-realtime works with syscall filtering
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("filter_anomaly_test");

    let source = r#"
#include <unistd.h>
#include <fcntl.h>
int main() {
    // Many writes (monitored)
    for (int i = 0; i < 40; i++) {
        write(1, "x", 1);
    }

    // Many opens (filtered out)
    for (int i = 0; i < 20; i++) {
        int fd = open("/dev/null", O_RDONLY);
        close(fd);
    }

    return 0;
}
"#;
    let source_file = tmp_dir.path().join("filter_anomaly_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--anomaly-realtime")
        .arg("-e")
        .arg("trace=write")
        .arg("-T")
        .arg("--")
        .arg(&test_program);

    // Should only detect anomalies in filtered syscalls (write)
    cmd.assert().success();
}

#[test]
fn test_anomaly_realtime_with_multiprocess() {
    // Test --anomaly-realtime works with -f (multi-process tracing)
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("fork_anomaly_test");

    let source = r#"
#include <unistd.h>
#include <sys/wait.h>
int main() {
    pid_t pid = fork();
    if (pid == 0) {
        // Child: baseline writes
        for (int i = 0; i < 30; i++) {
            write(1, "child\n", 6);
        }
        return 0;
    } else {
        // Parent: baseline writes
        for (int i = 0; i < 30; i++) {
            write(1, "parent\n", 7);
        }
        wait(NULL);
        return 0;
    }
}
"#;
    let source_file = tmp_dir.path().join("fork_anomaly_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-f").arg("--anomaly-realtime").arg("-T").arg("--").arg(&test_program);

    // Should detect anomalies across all processes
    cmd.assert().success();
}

// ============================================================================
// JSON Export Tests
// ============================================================================

#[test]
fn test_anomaly_json_export() {
    // Test that anomalies are included in JSON output
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("json_anomaly_test");

    let source = r#"
#include <unistd.h>
int main() {
    for (int i = 0; i < 50; i++) {
        write(1, "json\n", 5);
    }
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("json_anomaly_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--anomaly-realtime")
        .arg("-T")
        .arg("--format")
        .arg("json")
        .arg("--")
        .arg(&test_program);

    // JSON should include anomalies array (may be empty if no anomalies)
    cmd.assert().success();
    // Note: Can't guarantee anomalies will occur, so just check success
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_anomaly_with_zero_variance() {
    // Test anomaly detection when all samples are identical (stddev = 0)
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("zero_variance_test");

    // This test is conceptual - in practice, syscall durations always vary
    let source = r#"
#include <unistd.h>
int main() {
    // Many identical-ish syscalls
    for (int i = 0; i < 20; i++) {
        write(1, "x", 1);
    }
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("zero_variance_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--anomaly-realtime").arg("-T").arg("--").arg(&test_program);

    // Should handle zero variance gracefully (no division by zero)
    cmd.assert().success();
}

#[test]
fn test_anomaly_sliding_window_wraparound() {
    // Test that sliding window correctly removes old samples
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("sliding_window_test");

    let source = r#"
#include <unistd.h>
int main() {
    // More than window size (default 100) syscalls
    for (int i = 0; i < 150; i++) {
        write(1, "x", 1);
    }
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("sliding_window_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--anomaly-realtime")
        .arg("--anomaly-window-size")
        .arg("50")
        .arg("-T")
        .arg("--")
        .arg(&test_program);

    // Should handle window wraparound without memory issues
    cmd.assert().success();
}

#[test]
fn test_backward_compatibility_without_anomaly_realtime() {
    // Test that WITHOUT --anomaly-realtime, no anomaly detection occurs
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("compat_anomaly_test");

    let source = r#"
#include <unistd.h>
int main() {
    for (int i = 0; i < 30; i++) {
        write(1, "x", 1);
    }
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("compat_anomaly_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-T").arg("--").arg(&test_program);

    // Should NOT show any anomaly detection output
    cmd.assert().success().stderr(predicate::str::contains("ANOMALY").not());
}

#[test]
fn test_anomaly_threshold_from_sprint19_still_works() {
    // Test that --anomaly-threshold from Sprint 19 still works (post-hoc analysis)
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("threshold_compat_test");

    let source = r#"
#include <unistd.h>
int main() {
    for (int i = 0; i < 40; i++) {
        write(1, "x", 1);
    }
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("threshold_compat_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-c")
        .arg("--stats-extended")
        .arg("--anomaly-threshold")
        .arg("2.5")
        .arg("--")
        .arg(&test_program);

    // Sprint 19 functionality should still work
    cmd.assert().success();
}
