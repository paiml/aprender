// Sprint 23: ML-Enhanced Anomaly Detection via Aprender
// EXTREME TDD: RED phase - Integration tests for ML-based anomaly detection

use predicates::prelude::*;
use std::fs;
use tempfile::TempDir;

// ============================================================================
// Basic ML Anomaly Detection Tests
// ============================================================================

#[test]
fn test_ml_anomaly_flag_accepted() {
    // Test that --ml-anomaly flag is recognized
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("ml_basic_test");

    let source = r#"
#include <unistd.h>
int main() {
    for (int i = 0; i < 30; i++) {
        write(1, "x", 1);
    }
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("ml_basic_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("Failed to compile test program");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--ml-anomaly").arg("-T").arg("--").arg(&test_program);

    // Should accept the flag and run ML analysis
    cmd.assert().success();
}

#[test]
fn test_ml_anomaly_produces_cluster_output() {
    // Test that ML anomaly detection produces cluster analysis output
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("ml_cluster_test");

    let source = r#"
#include <unistd.h>
#include <fcntl.h>
int main() {
    // Mix of syscalls to create interesting clusters
    for (int i = 0; i < 20; i++) {
        write(1, "data\n", 5);
    }
    for (int i = 0; i < 10; i++) {
        int fd = open("/dev/null", O_RDONLY);
        close(fd);
    }
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("ml_cluster_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--ml-anomaly").arg("-c").arg("-T").arg("--").arg(&test_program);

    // Should show ML cluster analysis in stderr
    cmd.assert().success().stderr(predicate::str::contains("ML Anomaly Detection"));
}

#[test]
fn test_ml_clusters_configuration() {
    // Test that --ml-clusters configures number of clusters
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("ml_clusters_test");

    let source = r#"
#include <unistd.h>
int main() {
    for (int i = 0; i < 50; i++) {
        write(1, "x", 1);
    }
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("ml_clusters_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--ml-anomaly").arg("--ml-clusters").arg("5").arg("-T").arg("--").arg(&test_program);

    // Should accept custom cluster count
    cmd.assert().success();
}

#[test]
fn test_ml_silhouette_score_output() {
    // Test that ML analysis outputs silhouette score for clustering quality
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("ml_silhouette_test");

    let source = r#"
#include <unistd.h>
#include <fcntl.h>
int main() {
    // Multiple syscall types for clustering
    for (int i = 0; i < 30; i++) {
        write(1, "x", 1);
    }
    for (int i = 0; i < 20; i++) {
        int fd = open("/dev/null", O_RDONLY);
        read(fd, NULL, 0);
        close(fd);
    }
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("ml_silhouette_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--ml-anomaly").arg("-c").arg("-T").arg("--").arg(&test_program);

    // Should show silhouette score in stderr
    cmd.assert().success().stderr(predicate::str::contains("Silhouette"));
}

#[test]
fn test_ml_compare_with_zscore() {
    // Test --ml-compare shows both ML and z-score results
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("ml_compare_test");

    let source = r#"
#include <unistd.h>
#include <time.h>
int main() {
    // Baseline: fast writes
    for (int i = 0; i < 50; i++) {
        write(1, "x", 1);
    }

    // Anomaly: slow write
    struct timespec ts = {0, 10000000};  // 10ms
    nanosleep(&ts, NULL);
    write(1, "slow", 4);

    return 0;
}
"#;
    let source_file = tmp_dir.path().join("ml_compare_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--ml-anomaly").arg("--ml-compare").arg("-c").arg("-T").arg("--").arg(&test_program);

    // Should show comparison between ML and z-score in stderr
    cmd.assert()
        .success()
        .stderr(predicate::str::contains("Z-Score").or(predicate::str::contains("Comparison")));
}

// ============================================================================
// Integration with Existing Flags
// ============================================================================

#[test]
fn test_ml_anomaly_with_statistics() {
    // Test --ml-anomaly works with -c statistics mode
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("ml_stats_test");

    let source = r#"
#include <unistd.h>
int main() {
    for (int i = 0; i < 40; i++) {
        write(1, "data\n", 5);
    }
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("ml_stats_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-c").arg("--ml-anomaly").arg("-T").arg("--").arg(&test_program);

    // Should show both statistics and ML analysis
    cmd.assert().success().stderr(predicate::str::contains("% time"));
}

#[test]
fn test_ml_anomaly_with_filtering() {
    // Test --ml-anomaly works with syscall filtering
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("ml_filter_test");

    let source = r#"
#include <unistd.h>
#include <fcntl.h>
int main() {
    for (int i = 0; i < 30; i++) {
        write(1, "x", 1);
    }
    for (int i = 0; i < 20; i++) {
        int fd = open("/dev/null", O_RDONLY);
        close(fd);
    }
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("ml_filter_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--ml-anomaly").arg("-e").arg("trace=write").arg("-T").arg("--").arg(&test_program);

    // Should only analyze filtered syscalls
    cmd.assert().success();
}

#[test]
fn test_ml_anomaly_with_multiprocess() {
    // Test --ml-anomaly works with -f (multi-process tracing)
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("ml_fork_test");

    let source = r#"
#include <unistd.h>
#include <sys/wait.h>
int main() {
    pid_t pid = fork();
    if (pid == 0) {
        for (int i = 0; i < 20; i++) {
            write(1, "child\n", 6);
        }
        return 0;
    } else {
        for (int i = 0; i < 20; i++) {
            write(1, "parent\n", 7);
        }
        wait(NULL);
        return 0;
    }
}
"#;
    let source_file = tmp_dir.path().join("ml_fork_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-f").arg("--ml-anomaly").arg("-T").arg("--").arg(&test_program);

    // Should analyze all processes together
    cmd.assert().success();
}

#[test]
fn test_ml_anomaly_with_json_output() {
    // Test ML results are included in JSON output
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("ml_json_test");

    let source = r#"
#include <unistd.h>
int main() {
    for (int i = 0; i < 30; i++) {
        write(1, "json\n", 5);
    }
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("ml_json_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--ml-anomaly").arg("-T").arg("--format").arg("json").arg("--").arg(&test_program);

    // JSON should include ml_analysis field
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("ml_analysis").or(predicate::str::contains("clusters")));
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

#[test]
fn test_ml_anomaly_insufficient_data() {
    // Test ML handles insufficient data gracefully
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("ml_insufficient_test");

    let source = r#"
#include <unistd.h>
int main() {
    // Only 3 syscalls - not enough for meaningful clustering
    write(1, "a", 1);
    write(1, "b", 1);
    write(1, "c", 1);
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("ml_insufficient_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--ml-anomaly").arg("-e").arg("trace=write").arg("-T").arg("--").arg(&test_program);

    // Should handle gracefully (no crash, maybe warning)
    cmd.assert().success();
}

#[test]
fn test_ml_clusters_invalid_value() {
    // Test invalid cluster count is rejected
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--ml-anomaly")
        .arg("--ml-clusters")
        .arg("1") // Invalid: must be >= 2
        .arg("--")
        .arg("true");

    // Should reject invalid cluster count
    cmd.assert().failure();
}

#[test]
fn test_ml_clusters_minimum_value() {
    // Test minimum cluster count (2)
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("ml_min_clusters_test");

    let source = r#"
#include <unistd.h>
int main() {
    for (int i = 0; i < 30; i++) {
        write(1, "x", 1);
    }
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("ml_min_clusters_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--ml-anomaly").arg("--ml-clusters").arg("2").arg("-T").arg("--").arg(&test_program);

    // Should work with minimum cluster count
    cmd.assert().success();
}

#[test]
fn test_backward_compatibility_without_ml_anomaly() {
    // Test that WITHOUT --ml-anomaly, no ML analysis occurs
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("ml_compat_test");

    let source = r#"
#include <unistd.h>
int main() {
    for (int i = 0; i < 20; i++) {
        write(1, "x", 1);
    }
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("ml_compat_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-c").arg("-T").arg("--").arg(&test_program);

    // Should NOT show any ML analysis output
    cmd.assert()
        .success()
        .stderr(predicate::str::contains("ML Anomaly").not())
        .stderr(predicate::str::contains("Silhouette").not());
}

// ============================================================================
// ML Algorithm Correctness Tests
// ============================================================================

#[test]
fn test_ml_detects_outlier_cluster() {
    // Test that ML clustering identifies outlier patterns
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("ml_outlier_test");

    let source = r#"
#include <unistd.h>
#include <time.h>
int main() {
    // Cluster 1: Fast writes (normal)
    for (int i = 0; i < 50; i++) {
        write(1, "x", 1);
    }

    // Cluster 2: Slow writes (anomalous)
    for (int i = 0; i < 5; i++) {
        struct timespec ts = {0, 5000000};  // 5ms
        nanosleep(&ts, NULL);
        write(1, "slow", 4);
    }

    return 0;
}
"#;
    let source_file = tmp_dir.path().join("ml_outlier_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--ml-anomaly").arg("-c").arg("-T").arg("--").arg(&test_program);

    // Should identify outlier cluster
    cmd.assert()
        .success()
        .stderr(predicate::str::contains("cluster").or(predicate::str::contains("Cluster")));
}

#[test]
fn test_ml_multiple_syscall_types() {
    // Test ML handles multiple syscall types correctly
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("ml_multi_syscall_test");

    let source = r#"
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
int main() {
    // Mix of different syscall types
    for (int i = 0; i < 20; i++) {
        write(1, "w", 1);
        int fd = open("/dev/null", O_RDONLY);
        struct stat st;
        fstat(fd, &st);
        close(fd);
    }
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("ml_multi_syscall_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--ml-anomaly").arg("-c").arg("-T").arg("--").arg(&test_program);

    // Should cluster multiple syscall types
    cmd.assert().success();
}

// ============================================================================
// Combined ML + Realtime Anomaly Detection
// ============================================================================

#[test]
fn test_ml_anomaly_with_realtime() {
    // Test ML can work alongside realtime anomaly detection
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("ml_realtime_test");

    let source = r#"
#include <unistd.h>
int main() {
    for (int i = 0; i < 50; i++) {
        write(1, "x", 1);
    }
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("ml_realtime_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--ml-anomaly").arg("--anomaly-realtime").arg("-T").arg("--").arg(&test_program);

    // Should support both methods simultaneously
    cmd.assert().success();
}
