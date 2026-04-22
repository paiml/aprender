//! `cgp bench` — Enhanced criterion benchmarking with hardware counters.
//! Spec section 2.3: run cargo bench, capture criterion output,
//! optionally overlay perf stat counters, check regression.

use anyhow::{Context, Result};
use std::process::Command;

/// Run the `cgp bench` command.
pub fn run_bench(
    bench_name: &str,
    counters: Option<&str>,
    check_regression: bool,
    threshold: f64,
    _roofline: bool,
) -> Result<()> {
    println!("\n=== CGP Bench: {bench_name} ===\n");
    print_bench_header(bench_name, counters, check_regression, threshold);

    let output = run_cargo_bench(bench_name)?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if !output.status.success() {
        handle_bench_failure(bench_name, &stderr);
        return Ok(());
    }

    print_bench_results(&stdout);
    run_perf_overlay_if_requested(bench_name, counters);
    if check_regression {
        print_regression_check(&stdout, threshold);
    }

    println!();
    Ok(())
}

/// Print the pre-run configuration banner (counters/regression/perf availability).
fn print_bench_header(
    bench_name: &str,
    counters: Option<&str>,
    check_regression: bool,
    threshold: f64,
) {
    if counters.is_some() && which::which("perf").is_ok() {
        println!("  Hardware counter overlay: enabled");
    }
    println!("  Running: cargo bench --bench {bench_name}");
    if let Some(c) = counters {
        println!("  Hardware counters: {c}");
    }
    if check_regression {
        println!("  Regression check: threshold={threshold}%");
    }
}

/// Invoke `cargo bench`, splitting `bench_name/filter` into a criterion filter if present.
fn run_cargo_bench(bench_name: &str) -> Result<std::process::Output> {
    let mut cmd = Command::new("cargo");
    cmd.arg("bench");
    if let Some((bench, filter)) = bench_name.split_once('/') {
        cmd.arg("--bench").arg(bench).arg("--").arg(filter);
    } else {
        cmd.arg("--bench").arg(bench_name);
    }
    cmd.arg("--no-fail-fast");
    cmd.output()
        .with_context(|| format!("Failed to run cargo bench --bench {bench_name}"))
}

/// Handle a non-zero exit from cargo bench: list benches if missing, otherwise print stderr.
fn handle_bench_failure(bench_name: &str, stderr: &str) {
    if stderr.contains("no bench target") || stderr.contains("can't find") {
        println!("  Benchmark '{bench_name}' not found.");
        list_available_benches();
        return;
    }
    eprintln!("  cargo bench failed:\n{stderr}");
}

/// Print cargo's list of available bench targets by triggering a deliberate miss.
fn list_available_benches() {
    println!("  Available benchmarks:");
    let Ok(lo) = Command::new("cargo")
        .args(["bench", "--bench", "nonexistent_xyz_123", "--", "--list"])
        .output()
    else {
        return;
    };
    let lo_stderr = String::from_utf8_lossy(&lo.stderr);
    for line in lo_stderr.lines() {
        if line.contains("bench target") || line.contains("available") {
            println!("    {line}");
        }
    }
}

/// Parse criterion output and print either structured results or a raw preview.
fn print_bench_results(stdout: &str) {
    let results = parse_bench_results(stdout);
    if results.is_empty() {
        println!("  Criterion output:");
        for line in stdout.lines().take(30) {
            if !line.trim().is_empty() {
                println!("  {line}");
            }
        }
        return;
    }
    println!("  Results:");
    for r in &results {
        let change_str = match &r.change {
            Some(c) => format!("  ({c})"),
            None => String::new(),
        };
        println!("    {:40} {}{}", r.name, r.timing, change_str);
    }
}

/// Extract `name — time: X ns (change: ±Y%)` triples from criterion stdout.
fn parse_bench_results(stdout: &str) -> Vec<BenchResult> {
    let mut results: Vec<BenchResult> = Vec::new();
    for line in stdout.lines() {
        if !line.contains("time:") {
            continue;
        }
        let parts: Vec<&str> = line.splitn(2, "time:").collect();
        if parts.len() != 2 {
            continue;
        }
        let name = parts[0].trim().to_string();
        let timing = parts[1].trim().to_string();
        let change = extract_change(line);
        results.push(BenchResult {
            name,
            timing,
            change,
        });
    }
    results
}

/// Run perf stat overlay if counters were requested; fall back to an informative message.
fn run_perf_overlay_if_requested(bench_name: &str, counters: Option<&str>) {
    let Some(counter_list) = counters else {
        return;
    };
    if which::which("perf").is_ok() {
        println!("\n  --- perf stat overlay ---");
        run_perf_overlay(bench_name, counter_list);
    } else {
        println!("\n  perf not available — skipping hardware counter overlay.");
        println!("  Install linux-tools-common for hardware counter support.");
    }
}

/// Scan stdout for criterion regression/improvement lines and print a summary.
fn print_regression_check(stdout: &str, threshold: f64) {
    println!("\n  Regression check (threshold={threshold}%):");
    let mut regressions = 0;
    for line in stdout.lines() {
        if line.contains("regressed") || line.contains("Performance has regressed") {
            println!("    \x1b[31mREGRESSION\x1b[0m: {line}");
            regressions += 1;
        } else if line.contains("improved") {
            println!("    \x1b[32mIMPROVED\x1b[0m: {line}");
        }
    }
    if regressions > 0 {
        println!("\n  \x1b[31m{regressions} regression(s) detected!\x1b[0m");
    } else {
        println!("  No regressions detected.");
    }
}

/// Benchmark result parsed from criterion output.
struct BenchResult {
    name: String,
    timing: String,
    change: Option<String>,
}

/// Extract performance change annotation from criterion line.
fn extract_change(line: &str) -> Option<String> {
    if line.contains("change:") {
        line.split("change:").nth(1).map(|s| s.trim().to_string())
    } else {
        None
    }
}

/// Run perf stat with specified counters alongside the benchmark.
fn run_perf_overlay(bench_name: &str, counters: &str) {
    let events = counters.replace(' ', "");

    // Build the perf stat command wrapping cargo bench
    let mut cmd = Command::new("perf");
    cmd.arg("stat")
        .arg("-e")
        .arg(&events)
        .arg("-x")
        .arg(",")
        .arg("cargo")
        .arg("bench")
        .arg("--bench")
        .arg(bench_name)
        .arg("--")
        .arg("--quick"); // Use quick mode for perf overlay

    match cmd.output() {
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);

            // Parse perf stat CSV output from stderr
            for line in stderr.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') || line.starts_with("Performance") {
                    continue;
                }
                if line.contains("seconds time elapsed") {
                    println!("    Wall time: {}", line.trim());
                    continue;
                }

                let fields: Vec<&str> = line.split(',').collect();
                if fields.len() >= 3 {
                    let value = fields[0].trim();
                    let event = fields[2].trim();
                    if !value.is_empty() && !event.is_empty() {
                        println!("    {event:40} {value:>14}");
                    }
                }
            }
        }
        Err(e) => {
            println!("    perf stat failed: {e}");
            println!("    Try: sudo sysctl kernel.perf_event_paranoid=2");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_change_none() {
        assert!(extract_change("some time: [1.0 ns 2.0 ns]").is_none());
    }

    #[test]
    fn test_extract_change_present() {
        let change = extract_change("some time: [1.0 ns] change: +5.2%");
        assert!(change.is_some());
        assert!(change.unwrap().contains("+5.2%"));
    }

    #[test]
    fn test_bench_result_struct() {
        let r = BenchResult {
            name: "test".to_string(),
            timing: "1.0 ns".to_string(),
            change: None,
        };
        assert_eq!(r.name, "test");
    }
}
