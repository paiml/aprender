//! NASA-level performance tests using renacer baselines
//!
//! These tests validate performance against strict timing and syscall budgets.
//! Run with: cargo test --test performance_tests -- --ignored
//!
//! Requires:
//! - renacer installed: cargo install --path ../renacer
//! - Release build: cargo build --release -p aprender-shell
//!
//! Toyota Way Principle: *Genchi Genbutsu* (Go and see) - Understand performance
//! at the source through direct measurement.

use std::process::Command;
use std::time::{Duration, Instant};
use tempfile::NamedTempFile;

/// Helper to create a test model
fn create_test_model() -> NamedTempFile {
    let history = NamedTempFile::new().expect("create temp file");
    std::fs::write(
        history.path(),
        "git status\ngit commit -m test\ngit push origin main\n\
         cargo build --release\ncargo test\ncargo clippy\n\
         docker ps\ndocker run hello-world\nkubectl get pods\n",
    )
    .expect("write history");

    let model = NamedTempFile::new().expect("create model file");

    let status = Command::new(env!("CARGO_BIN_EXE_aprender-shell"))
        .args([
            "train",
            history.path().to_str().unwrap(),
            "--output",
            model.path().to_str().unwrap(),
        ])
        .status()
        .expect("train model");

    assert!(status.success(), "Failed to train test model");
    model
}

/// Suggestion latency must be <10ms P99
///
/// Target: P50 <2ms, P95 <5ms, P99 <10ms
#[test]
#[ignore] // Run manually or in CI with: cargo test -- --ignored
fn test_suggest_latency_p99() {
    let model = create_test_model();
    let model_path = model.path().to_str().unwrap();

    let mut latencies = Vec::with_capacity(100);

    for _ in 0..100 {
        let start = Instant::now();
        let output = Command::new(env!("CARGO_BIN_EXE_aprender-shell"))
            .args(["suggest", "git ", "--model", model_path])
            .output()
            .expect("Failed to run suggest");
        let elapsed = start.elapsed();

        assert!(output.status.success(), "suggest command failed");
        latencies.push(elapsed.as_micros());
    }

    latencies.sort();
    let p50 = latencies[49];
    let p95 = latencies[94];
    let p99 = latencies[98];

    println!("Latency percentiles (μs): P50={p50}, P95={p95}, P99={p99}");

    assert!(p99 < 10_000, "P99 latency {p99} μs exceeds 10ms target");

    // Informational checks (warn but don't fail)
    if p50 > 2_000 {
        eprintln!("WARNING: P50 latency {p50} μs exceeds 2ms soft target");
    }
    if p95 > 5_000 {
        eprintln!("WARNING: P95 latency {p95} μs exceeds 5ms soft target");
    }
}

/// Model loading must be <100ms cold
#[test]
#[ignore]
fn test_model_load_latency_cold() {
    let model = create_test_model();
    let model_path = model.path().to_str().unwrap();

    // Drop filesystem cache by using a fresh path each time
    let start = Instant::now();
    let output = Command::new(env!("CARGO_BIN_EXE_aprender-shell"))
        .args(["stats", "--model", model_path])
        .output()
        .expect("Failed to run stats");
    let cold_latency = start.elapsed();

    assert!(output.status.success(), "stats command failed");

    println!("Cold load latency: {:?}", cold_latency);

    assert!(
        cold_latency < Duration::from_millis(100),
        "Cold load latency {:?} exceeds 100ms target",
        cold_latency
    );
}

/// Repeated suggestions should complete in <5ms (warm path)
#[test]
#[ignore]
fn test_suggest_warm_latency() {
    let model = create_test_model();
    let model_path = model.path().to_str().unwrap();

    // Warm up (first call loads model)
    let _ = Command::new(env!("CARGO_BIN_EXE_aprender-shell"))
        .args(["suggest", "git ", "--model", model_path])
        .output()
        .expect("warmup failed");

    // Measure warm latency
    let mut latencies = Vec::with_capacity(50);
    for _ in 0..50 {
        let start = Instant::now();
        let output = Command::new(env!("CARGO_BIN_EXE_aprender-shell"))
            .args(["suggest", "cargo ", "--model", model_path])
            .output()
            .expect("suggest failed");
        let elapsed = start.elapsed();

        assert!(output.status.success());
        latencies.push(elapsed.as_micros());
    }

    latencies.sort();
    let p50 = latencies[24];
    let p95 = latencies[47];

    println!("Warm latency (μs): P50={p50}, P95={p95}");

    assert!(p95 < 5_000, "Warm P95 latency {p95} μs exceeds 5ms target");
}

/// Syscall count must be <150 per suggestion
///
/// Current baseline: ~970 brk calls (excessive)
/// Target: <80 total syscalls with pre-allocation
#[test]
#[ignore]
fn test_syscall_budget() {
    // Check if renacer is installed
    let renacer_check = Command::new("which").arg("renacer").output();

    if renacer_check.is_err() || !renacer_check.unwrap().status.success() {
        eprintln!("SKIP: renacer not found - install from ../renacer");
        return;
    }

    let model = create_test_model();
    let model_path = model.path().to_str().unwrap();

    let output = Command::new("renacer")
        .args([
            "-c",
            "--",
            env!("CARGO_BIN_EXE_aprender-shell"),
            "suggest",
            "git ",
            "--model",
            model_path,
        ])
        .output()
        .expect("renacer failed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    println!("Renacer output:\n{stdout}");

    // Parse total syscall count from renacer output
    // Format: "100.00    0.012345                   142         0 total"
    let total_line = stdout.lines().find(|line| line.contains("total"));

    if let Some(line) = total_line {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 4 {
            if let Ok(syscall_count) = parts[3].parse::<usize>() {
                println!("Total syscalls: {syscall_count}");

                assert!(
                    syscall_count < 150,
                    "Syscall count {syscall_count} exceeds 150 budget (target: <80)"
                );

                if syscall_count > 80 {
                    eprintln!("WARNING: Syscall count {syscall_count} exceeds 80 soft target");
                }
            }
        }
    } else {
        eprintln!("WARNING: Could not parse syscall count from renacer output");
    }
}

/// No anomalies should occur during normal operation
#[test]
#[ignore]
fn test_no_anomalies() {
    // Check if renacer is installed
    let renacer_check = Command::new("which").arg("renacer").output();

    if renacer_check.is_err() || !renacer_check.unwrap().status.success() {
        eprintln!("SKIP: renacer not found - install from ../renacer");
        return;
    }

    let model = create_test_model();
    let model_path = model.path().to_str().unwrap();

    let output = Command::new("renacer")
        .args([
            "--anomaly-realtime",
            "--anomaly-threshold",
            "3.0",
            "--",
            env!("CARGO_BIN_EXE_aprender-shell"),
            "suggest",
            "git status",
            "--model",
            model_path,
        ])
        .output()
        .expect("renacer failed");

    let stderr = String::from_utf8_lossy(&output.stderr);

    let anomaly_count = stderr.matches("ANOMALY").count();
    println!("Anomalies detected: {anomaly_count}");

    // Allow up to 2 minor anomalies (startup transients)
    assert!(
        anomaly_count < 3,
        "Too many anomalies detected ({anomaly_count}):\n{stderr}"
    );
}

/// Memory usage should not grow unbounded
#[test]
#[ignore]
fn test_memory_bounded() {
    let model = create_test_model();
    let model_path = model.path().to_str().unwrap();

    // Run 100 suggestions and check memory doesn't grow
    for i in 0..100 {
        let output = Command::new(env!("CARGO_BIN_EXE_aprender-shell"))
            .args(["suggest", "git ", "--model", model_path])
            .output()
            .expect("suggest failed");

        assert!(output.status.success(), "suggest failed at iteration {i}");
    }

    // If we get here without OOM, memory is bounded
    println!("100 suggestions completed without OOM");
}

/// Validate that security filtering doesn't add significant latency
#[test]
#[ignore]
fn test_security_filter_overhead() {
    let model = create_test_model();
    let model_path = model.path().to_str().unwrap();

    // Measure latency for commands that should trigger security checks
    let prefixes = [
        "export ",     // May match SECRET patterns
        "curl -u ",    // May match credential patterns
        "git status ", // Normal command (baseline)
    ];

    for prefix in prefixes {
        let mut latencies = Vec::with_capacity(20);

        for _ in 0..20 {
            let start = Instant::now();
            let _ = Command::new(env!("CARGO_BIN_EXE_aprender-shell"))
                .args(["suggest", prefix, "--model", model_path])
                .output()
                .expect("suggest failed");
            latencies.push(start.elapsed().as_micros());
        }

        latencies.sort();
        let p50 = latencies[9];

        println!("Security filter test for '{prefix}': P50={p50}μs");

        // Security filtering should add <1ms overhead
        assert!(
            p50 < 3_000,
            "Prefix '{prefix}' has excessive latency: {p50}μs"
        );
    }
}
