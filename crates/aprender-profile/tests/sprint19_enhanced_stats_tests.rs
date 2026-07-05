// Sprint 19: Enhanced Statistics with Trueno
// EXTREME TDD: RED phase - Integration tests for advanced statistical analysis

use predicates::prelude::*;
use std::fs;
use tempfile::TempDir;

// ============================================================================
// Basic Enhanced Statistics Tests
// ============================================================================

#[test]
fn test_stats_extended_calculates_percentiles() {
    // Test that --stats-extended flag shows percentile calculations
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("stats_test");

    // Create a program with many write syscalls to get statistical data
    let source = r#"
#include <unistd.h>
int main() {
    for (int i = 0; i < 100; i++) {
        write(1, "x", 1);
    }
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("stats_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("Failed to compile test program");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-c").arg("--stats-extended").arg("--").arg(&test_program);

    // Should show enhanced statistics with percentiles
    cmd.assert()
        .success()
        .stderr(predicate::str::contains("Mean:"))
        .stderr(predicate::str::contains("Std Dev:"))
        .stderr(predicate::str::contains("Median (P50):"))
        .stderr(predicate::str::contains("P95:"))
        .stderr(predicate::str::contains("P99:"));
}

#[test]
fn test_stats_extended_shows_min_max() {
    // Test that --stats-extended shows min/max durations
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("minmax_test");

    let source = r#"
#include <unistd.h>
#include <fcntl.h>
int main() {
    // Mix of fast and slow syscalls
    for (int i = 0; i < 50; i++) {
        write(1, "x", 1);  // Fast
    }
    int fd = open("/dev/null", O_RDONLY);  // Potentially slower
    close(fd);
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("minmax_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-c").arg("--stats-extended").arg("-T").arg("--").arg(&test_program);

    cmd.assert()
        .success()
        .stderr(predicate::str::contains("Min:"))
        .stderr(predicate::str::contains("Max:"))
        .stderr(predicate::str::contains("μs"));
}

#[test]
fn test_stats_extended_with_timing() {
    // Test --stats-extended works with -T timing mode
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("timing_test");

    let source = r#"
#include <unistd.h>
int main() {
    for (int i = 0; i < 20; i++) {
        write(1, "test\n", 5);
    }
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("timing_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-c").arg("-T").arg("--stats-extended").arg("--").arg(&test_program);

    // Should show timing statistics with percentiles
    cmd.assert()
        .success()
        .stderr(predicate::str::contains("% time"))
        .stderr(predicate::str::contains("Mean:"))
        .stderr(predicate::str::contains("P95:"));
}

// ============================================================================
// Anomaly Detection Tests
// ============================================================================

#[test]
fn test_anomaly_detection_slow_syscall() {
    // Test that unusually slow syscalls are flagged as anomalies
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("anomaly_test");

    let source = r#"
#include <unistd.h>
#include <fcntl.h>
int main() {
    // Many fast writes to establish baseline
    for (int i = 0; i < 100; i++) {
        write(1, "x", 1);
    }

    // Potentially slower syscall (disk I/O)
    sync();

    return 0;
}
"#;
    let source_file = tmp_dir.path().join("anomaly_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-c").arg("--stats-extended").arg("-T").arg("--").arg(&test_program);

    // Should detect and flag anomalies
    cmd.assert().success();
    // Note: May or may not show anomaly depending on system load
    // This test primarily ensures no crash when checking for anomalies
}

#[test]
fn test_anomaly_threshold_configuration() {
    // Test that anomaly detection can be configured with custom threshold
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("threshold_test");

    let source = r#"
#include <unistd.h>
int main() {
    for (int i = 0; i < 50; i++) {
        write(1, "x", 1);
    }
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("threshold_test.c");
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

    // Should accept custom threshold (2.5 sigma instead of default 3.0)
    cmd.assert().success();
}

// ============================================================================
// Integration with Other Flags
// ============================================================================

#[test]
fn test_stats_extended_with_filtering() {
    // Test --stats-extended works with syscall filtering
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("filter_stats_test");

    let source = r#"
#include <unistd.h>
#include <fcntl.h>
int main() {
    for (int i = 0; i < 30; i++) {
        write(1, "data", 4);
    }
    int fd = open("/dev/null", O_RDONLY);
    for (int i = 0; i < 20; i++) {
        read(fd, NULL, 0);
    }
    close(fd);
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("filter_stats_test.c");
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
        .arg("-e")
        .arg("trace=write,read")
        .arg("--")
        .arg(&test_program);

    // Should show extended stats only for filtered syscalls
    cmd.assert()
        .success()
        .stderr(predicate::str::contains("Mean:"))
        .stderr(predicate::str::contains("write"))
        .stderr(predicate::str::contains("read"));
}

#[test]
fn test_stats_extended_with_multiprocess() {
    // Test --stats-extended works with -f (multi-process tracing)
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("fork_stats_test");

    let source = r#"
#include <unistd.h>
#include <sys/wait.h>
int main() {
    pid_t pid = fork();
    if (pid == 0) {
        // Child: some writes
        for (int i = 0; i < 20; i++) {
            write(1, "child\n", 6);
        }
        return 0;
    } else {
        // Parent: some writes
        for (int i = 0; i < 30; i++) {
            write(1, "parent\n", 7);
        }
        wait(NULL);
        return 0;
    }
}
"#;
    let source_file = tmp_dir.path().join("fork_stats_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-f").arg("-c").arg("--stats-extended").arg("--").arg(&test_program);

    // Should show aggregated statistics across all processes
    cmd.assert()
        .success()
        .stderr(predicate::str::contains("Mean:"))
        .stderr(predicate::str::contains("% time"));
}

#[test]
fn test_stats_extended_json_output() {
    // Test --stats-extended with JSON output shows extended stats in stderr
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("json_stats_test");

    let source = r#"
#include <unistd.h>
int main() {
    for (int i = 0; i < 25; i++) {
        write(1, "json\n", 5);
    }
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("json_stats_test.c");
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
        .arg("-T")
        .arg("--format")
        .arg("json")
        .arg("--")
        .arg(&test_program);

    // JSON output goes to stdout, extended stats summary goes to stderr
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("\"syscalls\""))
        .stderr(predicate::str::contains("Mean:"))
        .stderr(predicate::str::contains("Std Dev:"))
        .stderr(predicate::str::contains("P95:"))
        .stderr(predicate::str::contains("P99:"));
}

#[test]
fn test_stats_extended_csv_output() {
    // Test --stats-extended with CSV output shows extended stats in stderr
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("csv_stats_test");

    let source = r#"
#include <unistd.h>
int main() {
    for (int i = 0; i < 25; i++) {
        write(1, "csv\n", 4);
    }
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("csv_stats_test.c");
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
        .arg("-T")
        .arg("--format")
        .arg("csv")
        .arg("--")
        .arg(&test_program);

    // CSV output goes to stdout, extended stats summary goes to stderr
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("syscall,calls,errors"))
        .stderr(predicate::str::contains("Mean:"))
        .stderr(predicate::str::contains("Std Dev:"))
        .stderr(predicate::str::contains("P95:"))
        .stderr(predicate::str::contains("P99:"));
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_stats_extended_single_syscall() {
    // Test extended stats with only one syscall (no variance)
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("single_test");

    let source = r#"
#include <unistd.h>
int main() {
    write(1, "single\n", 7);
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("single_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-c").arg("--stats-extended").arg("-T").arg("--").arg(&test_program);

    // Should handle single data point gracefully (stddev = 0)
    cmd.assert().success().stderr(predicate::str::contains("Mean:"));
}

#[test]
fn test_stats_extended_no_timing_data() {
    // Test --stats-extended without -T flag (no duration data)
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("no_timing_test");

    let source = r#"
#include <unistd.h>
int main() {
    for (int i = 0; i < 10; i++) {
        write(1, "x", 1);
    }
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("no_timing_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-c").arg("--stats-extended").arg("--").arg(&test_program);

    // Should show count statistics but skip duration percentiles
    cmd.assert().success().stderr(predicate::str::contains("% time"));
}

#[test]
fn test_stats_extended_large_dataset() {
    // Test extended stats with large number of syscalls
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("large_test");

    let source = r#"
#include <unistd.h>
int main() {
    for (int i = 0; i < 1000; i++) {
        write(1, "x", 1);
    }
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("large_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-c").arg("--stats-extended").arg("-T").arg("--").arg(&test_program);

    // Should handle large datasets efficiently with Trueno SIMD
    cmd.assert()
        .success()
        .stderr(predicate::str::contains("Mean:"))
        .stderr(predicate::str::contains("P99:"));
}

#[test]
fn test_backward_compatibility_without_stats_extended() {
    // Test that WITHOUT --stats-extended, output matches v0.3.0 behavior
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("compat_test");

    let source = r#"
#include <unistd.h>
int main() {
    for (int i = 0; i < 20; i++) {
        write(1, "x", 1);
    }
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
    cmd.arg("-c").arg("--").arg(&test_program);

    // Should NOT show extended statistics without flag
    cmd.assert()
        .success()
        .stderr(predicate::str::contains("% time"))
        .stderr(predicate::str::contains("Mean:").not())
        .stderr(predicate::str::contains("P95:").not());
}
