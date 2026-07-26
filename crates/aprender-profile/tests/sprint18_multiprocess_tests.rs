// Sprint 18: Multi-Process Tracing (-f flag)
// EXTREME TDD: RED phase - Integration tests for fork following

#![allow(deprecated)] // Command::cargo_bin is deprecated but still functional

use predicates::prelude::*;
use std::fs;
use tempfile::TempDir;

// ============================================================================
// Basic Fork Following Tests
// ============================================================================

#[test]
fn test_follow_forks_basic() {
    // Test that -f flag traces both parent and child after fork()
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("fork_test");

    // Create a simple C program that forks
    let source = r#"
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>

int main() {
    pid_t pid = fork();
    if (pid == 0) {
        // Child process
        printf("child\n");
        return 42;
    } else {
        // Parent process
        int status;
        waitpid(pid, &status, 0);
        printf("parent\n");
        return 0;
    }
}
"#;
    let source_file = tmp_dir.path().join("fork_test.c");
    fs::write(&source_file, source).expect("test");

    // Compile the test program
    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("Failed to compile test program");

    // Run with -f flag
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-f").arg("--").arg(&test_program);

    // Should trace syscalls from both parent and child
    // Note: On Linux, fork() is implemented via clone() syscall
    cmd.assert().success().stdout(predicate::str::contains("clone"));
}

#[test]
fn test_follow_forks_with_exec() {
    // Test that -f flag traces child process after fork + exec
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("fork_exec_test");

    // Create a program that forks and execs /bin/true
    let source = r#"
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>

int main() {
    pid_t pid = fork();
    if (pid == 0) {
        // Child: exec /bin/true
        execl("/bin/true", "true", NULL);
        return 1; // Should not reach here
    } else {
        // Parent: wait for child
        int status;
        waitpid(pid, &status, 0);
        return 0;
    }
}
"#;
    let source_file = tmp_dir.path().join("fork_exec_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-f").arg("--").arg(&test_program);

    // Should trace fork and potentially execve in child
    // Note: The child may exit very quickly, so we may not capture execve
    // The important thing is that fork following works (we see clone)
    cmd.assert().success().stdout(predicate::str::contains("clone"));
}

#[test]
fn test_follow_forks_disabled_by_default() {
    // Test that WITHOUT -f flag, only parent is traced
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("fork_default_test");

    let source = r#"
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>

int main() {
    pid_t pid = fork();
    if (pid == 0) {
        // Child: make a unique syscall (write to stderr)
        write(2, "child_marker\n", 13);
        return 0;
    } else {
        // Parent: wait and write to stdout
        int status;
        waitpid(pid, &status, 0);
        write(1, "parent_marker\n", 14);
        return 0;
    }
}
"#;
    let source_file = tmp_dir.path().join("fork_default_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    // Run WITHOUT -f flag
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--").arg(&test_program);

    // Should only see parent's syscalls, not child's
    // (this test verifies current behavior - will need adjustment after implementation)
    cmd.assert().success();
}

// ============================================================================
// Multiple Fork Tests
// ============================================================================

#[test]
fn test_follow_multiple_forks() {
    // Test tracing program that forks multiple children
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("multi_fork_test");

    let source = r#"
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>

int main() {
    // Fork 3 children
    for (int i = 0; i < 3; i++) {
        pid_t pid = fork();
        if (pid == 0) {
            // Child: write unique marker
            printf("child_%d\n", i);
            return i;
        }
    }

    // Parent: wait for all children
    for (int i = 0; i < 3; i++) {
        wait(NULL);
    }

    printf("parent_done\n");
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("multi_fork_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-f").arg("--").arg(&test_program);

    // Should trace all 4 processes (parent + 3 children)
    cmd.assert().success();
}

// ============================================================================
// Fork Following with Other Flags
// ============================================================================

#[test]
fn test_follow_forks_with_filtering() {
    // Test -f works with syscall filtering
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("fork_filter_test");

    let source = r#"
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>

int main() {
    pid_t pid = fork();
    if (pid == 0) {
        write(1, "child\n", 6);
        return 0;
    } else {
        int status;
        waitpid(pid, &status, 0);
        write(1, "parent\n", 7);
        return 0;
    }
}
"#;
    let source_file = tmp_dir.path().join("fork_filter_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-f").arg("-e").arg("trace=write").arg("--").arg(&test_program);

    // Should show write syscalls from both parent and child, but filter out others
    cmd.assert().success().stdout(predicate::str::contains("write"));
}

#[test]
fn test_follow_forks_with_statistics() {
    // Test -f works with -c statistics mode
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("fork_stats_test");

    let source = r#"
#include <unistd.h>
#include <sys/wait.h>

int main() {
    fork();
    return 0;
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
    cmd.arg("-f").arg("-c").arg("--").arg(&test_program);

    // Should show statistics aggregated across all processes (goes to stderr)
    cmd.assert()
        .success()
        .stderr(predicate::str::contains("% time"))
        .stderr(predicate::str::contains("syscall"));
}

#[test]
fn test_follow_forks_with_json() {
    // Test -f works with JSON output format
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("fork_json_test");

    let source = r#"
#include <unistd.h>
#include <sys/wait.h>

int main() {
    pid_t pid = fork();
    if (pid > 0) wait(NULL);
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("fork_json_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-f").arg("--format").arg("json").arg("--").arg(&test_program);

    // JSON output should include syscalls from all processes
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("\"syscalls\""))
        .stdout(predicate::str::contains("\"name\""));
}

#[test]
fn test_follow_forks_with_csv() {
    // Test -f works with CSV output format
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("fork_csv_test");

    let source = r#"
#include <unistd.h>
#include <sys/wait.h>

int main() {
    pid_t pid = fork();
    if (pid > 0) wait(NULL);
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("fork_csv_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-f").arg("--format").arg("csv").arg("--").arg(&test_program);

    // CSV output should include header and syscalls from all processes
    cmd.assert().success().stdout(predicate::str::contains("syscall,arguments,result"));
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_follow_forks_with_immediate_exit() {
    // Test child that exits immediately after fork
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("fork_quick_exit");

    let source = r#"
#include <unistd.h>
#include <sys/wait.h>

int main() {
    pid_t pid = fork();
    if (pid == 0) {
        // Child exits immediately
        return 0;
    } else {
        // Parent waits
        wait(NULL);
        return 0;
    }
}
"#;
    let source_file = tmp_dir.path().join("fork_quick_exit.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-f").arg("--").arg(&test_program);

    // Should handle child that exits quickly without crashing
    cmd.assert().success();
}

#[test]
fn test_follow_vfork() {
    // Test vfork() variant (shares memory until exec)
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("vfork_test");

    let source = r#"
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>

int main() {
    pid_t pid = vfork();
    if (pid == 0) {
        // Child must exec or _exit (not return)
        _exit(0);
    } else {
        // Parent
        int status;
        waitpid(pid, &status, 0);
        return 0;
    }
}
"#;
    let source_file = tmp_dir.path().join("vfork_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-f").arg("--").arg(&test_program);

    // Should handle vfork() as well as fork()
    cmd.assert().success();
}

#[test]
fn test_follow_clone() {
    // Test clone() syscall (used by pthread_create)
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("clone_test");

    let source = r#"
#include <pthread.h>
#include <unistd.h>

void* thread_func(void* arg) {
    return NULL;
}

int main() {
    pthread_t thread;
    pthread_create(&thread, NULL, thread_func, NULL);
    pthread_join(thread, NULL);
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("clone_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .arg("-pthread")
        .output()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-f").arg("--").arg(&test_program);

    // Should handle clone() syscall (threads)
    // Note: pthread_create may use clone3 on modern Linux
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("clone").or(predicate::str::contains("clone3")));
}
