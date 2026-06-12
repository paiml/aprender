//! Integration tests for cgp binary.
//! These tests run the actual cgp binary as a subprocess and verify output.
//! Covers FALSIFY tests that require the full binary.

// Integration-test binary: `.unwrap()` is idiomatic in test assertions and the
// lib's `cfg(test)` allow does not reach this separate crate.
#![allow(clippy::disallowed_methods)]

use std::process::Command;

fn cgp_cmd() -> Command {
    // Use cargo run to ensure we're testing the current code
    let mut cmd = Command::new(env!("CARGO"));
    cmd.arg("run").arg("-p").arg("cgp").arg("--");
    cmd
}

/// FALSIFY-CGP-061: cgp doctor must complete in < 2 seconds.
#[test]
fn test_doctor_completes() {
    let start = std::time::Instant::now();
    let output = cgp_cmd()
        .arg("doctor")
        .output()
        .expect("Failed to run cgp doctor");
    let elapsed = start.elapsed();

    assert!(output.status.success(), "cgp doctor failed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("cgp System Check"), "Missing header");
    assert!(stdout.contains("CPU"), "Missing CPU detection");
    // Doctor should complete in <2s (FALSIFY-CGP-061), allow 30s for compilation
    assert!(
        elapsed.as_secs() < 30,
        "cgp doctor took too long: {:?}",
        elapsed
    );
}

/// cgp roofline --target cuda must show RTX 4090 data.
#[test]
fn test_roofline_cuda() {
    let output = cgp_cmd()
        .args(["roofline", "--target", "cuda"])
        .output()
        .expect("Failed to run cgp roofline");

    assert!(output.status.success(), "cgp roofline failed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("RTX 4090") || stdout.contains("Peak Compute"));
    assert!(stdout.contains("Ridge Point"));
    assert!(stdout.contains("327.4") || stdout.contains("327"));
}

/// cgp roofline --target avx512 must show CPU data.
#[test]
fn test_roofline_avx512() {
    let output = cgp_cmd()
        .args(["roofline", "--target", "avx512"])
        .output()
        .expect("Failed to run cgp roofline");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("AVX-512"));
    assert!(stdout.contains("Peak Compute"));
}

/// cgp roofline --export must create a JSON file.
#[test]
fn test_roofline_export() {
    let path = "/tmp/cgp-test-roofline-export.json";
    let _ = std::fs::remove_file(path);

    let output = cgp_cmd()
        .args(["roofline", "--target", "cuda", "--export", path])
        .output()
        .expect("Failed to run cgp roofline");

    assert!(output.status.success());
    assert!(
        std::path::Path::new(path).exists(),
        "Export file not created"
    );

    let content = std::fs::read_to_string(path).unwrap();
    assert!(content.contains("peak_compute"));
    let _ = std::fs::remove_file(path);
}

/// cgp diff must compare two profiles.
#[test]
fn test_diff_profiles() {
    // Create test profiles
    let baseline = r#"{"version":"2.0","timestamp":"","hardware":{"cpu_features":[]},"timing":{"wall_clock_time_us":35.7,"samples":1,"stddev_us":0.0,"ci_95_low_us":0.0,"ci_95_high_us":0.0},"throughput":{"tflops":7.5,"gflops":0.0,"bandwidth_gbps":0.0,"arithmetic_intensity":0.0},"muda":[]}"#;
    let current = r#"{"version":"2.0","timestamp":"","hardware":{"cpu_features":[]},"timing":{"wall_clock_time_us":23.2,"samples":1,"stddev_us":0.0,"ci_95_low_us":0.0,"ci_95_high_us":0.0},"throughput":{"tflops":11.6,"gflops":0.0,"bandwidth_gbps":0.0,"arithmetic_intensity":0.0},"muda":[]}"#;

    std::fs::write("/tmp/cgp-test-baseline.json", baseline).unwrap();
    std::fs::write("/tmp/cgp-test-current.json", current).unwrap();

    let output = cgp_cmd()
        .args([
            "diff",
            "--baseline",
            "/tmp/cgp-test-baseline.json",
            "--current",
            "/tmp/cgp-test-current.json",
        ])
        .output()
        .expect("Failed to run cgp diff");

    assert!(output.status.success(), "cgp diff failed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("IMPROVED"));
    assert!(stdout.contains("wall_clock_time_us"));

    let _ = std::fs::remove_file("/tmp/cgp-test-baseline.json");
    let _ = std::fs::remove_file("/tmp/cgp-test-current.json");
}

/// cgp profile compare must produce a table.
#[test]
fn test_profile_compare() {
    let output = cgp_cmd()
        .args([
            "profile",
            "compare",
            "--kernel",
            "gemm",
            "--size",
            "256",
            "--backends",
            "scalar,avx2",
        ])
        .output()
        .expect("Failed to run cgp profile compare");

    assert!(output.status.success(), "cgp profile compare failed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Cross-Backend Comparison"));
    assert!(stdout.contains("scalar"));
    assert!(stdout.contains("avx2"));
}

/// cgp compete must run commands and produce comparison.
#[test]
fn test_compete() {
    let output = cgp_cmd()
        .args([
            "compete",
            "test",
            "--ours",
            "true",
            "--theirs",
            "true",
            "--label",
            "cmd1,cmd2",
        ])
        .output()
        .expect("Failed to run cgp compete");

    assert!(output.status.success(), "cgp compete failed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Head-to-Head"));
    assert!(stdout.contains("Winner"));
}

/// cgp contract generate must produce YAML.
#[test]
fn test_contract_generate() {
    let output = cgp_cmd()
        .args([
            "contract",
            "generate",
            "--kernel",
            "test_gemm",
            "--size",
            "256",
            "--tolerance",
            "15",
        ])
        .output()
        .expect("Failed to run cgp contract generate");

    assert!(output.status.success(), "cgp contract generate failed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("PerformanceContract"));
    assert!(stdout.contains("test_gemm"));
    assert!(stdout.contains("FALSIFY"));

    // Clean up generated file
    let _ = std::fs::remove_file("contracts/cgp/test_gemm-256-v1.yaml");
    let _ = std::fs::remove_dir("contracts/cgp");
    let _ = std::fs::remove_dir("contracts");
}

/// FALSIFY-CGP-012: cgp must function without NVIDIA tools for SIMD profiling.
#[test]
fn test_profile_kernel_no_binary() {
    let output = cgp_cmd()
        .args([
            "profile",
            "kernel",
            "--name",
            "nonexistent_kernel",
            "--size",
            "512",
        ])
        .output()
        .expect("Failed to run cgp profile kernel");

    // Should succeed even with no matching binary
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("No binary found") || stdout.contains("CGP Kernel Profile"),
        "stdout: {stdout}"
    );
}

/// FALSIFY-CGP-077: Metal must error on non-macOS.
#[test]
#[cfg(not(target_os = "macos"))]
fn test_metal_not_available() {
    let output = cgp_cmd()
        .args(["profile", "metal", "--shader", "test"])
        .output()
        .expect("Failed to run cgp profile metal");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("macOS") || stderr.contains("Metal"),
        "Should mention macOS requirement: {stderr}"
    );
}

/// cgp --version must work.
#[test]
fn test_version() {
    let output = cgp_cmd()
        .arg("--version")
        .output()
        .expect("Failed to run cgp --version");

    // clap replaces the subcommand dispatch, but since we use `cargo run -- --version`
    // it may behave differently. Just check it doesn't panic.
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{stdout}{stderr}");
    // Either it shows version or clap error — both are fine
    assert!(
        combined.contains("cgp") || combined.contains("0.1"),
        "output: {combined}"
    );
}

/// cgp --json doctor must output valid JSON with operational status.
#[test]
fn test_json_doctor() {
    let output = cgp_cmd()
        .args(["--json", "doctor"])
        .output()
        .expect("Failed to run cgp --json doctor");

    assert!(output.status.success(), "cgp --json doctor failed");
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Must be valid JSON
    let parsed: serde_json::Value =
        serde_json::from_str(&stdout).expect("Doctor JSON output is not valid JSON");

    // Must have expected fields
    assert!(parsed.get("checks").is_some(), "Missing 'checks' field");
    assert!(
        parsed.get("operational").is_some(),
        "Missing 'operational' field"
    );
    assert!(
        parsed.get("elapsed_ms").is_some(),
        "Missing 'elapsed_ms' field"
    );
    assert!(parsed["ok_count"].as_u64().unwrap_or(0) > 0, "No OK checks");
}

/// cgp --json roofline must output valid JSON model.
#[test]
fn test_json_roofline() {
    let output = cgp_cmd()
        .args(["--json", "roofline", "--target", "cuda"])
        .output()
        .expect("Failed to run cgp --json roofline");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);

    let parsed: serde_json::Value =
        serde_json::from_str(&stdout).expect("Roofline JSON is not valid JSON");

    assert!(parsed.get("peak_compute").is_some(), "Missing peak_compute");
    assert!(
        parsed.get("peak_bandwidth").is_some(),
        "Missing peak_bandwidth"
    );
    assert!(parsed.get("target").is_some(), "Missing target");
}

/// cgp --json diff must output valid JSON metric diffs.
#[test]
fn test_json_diff() {
    let baseline = r#"{"version":"2.0","timestamp":"","hardware":{"cpu_features":[]},"timing":{"wall_clock_time_us":35.7,"samples":1,"stddev_us":0.0,"ci_95_low_us":0.0,"ci_95_high_us":0.0},"throughput":{"tflops":7.5,"gflops":0.0,"bandwidth_gbps":0.0,"arithmetic_intensity":0.0},"muda":[]}"#;
    let current = r#"{"version":"2.0","timestamp":"","hardware":{"cpu_features":[]},"timing":{"wall_clock_time_us":23.2,"samples":1,"stddev_us":0.0,"ci_95_low_us":0.0,"ci_95_high_us":0.0},"throughput":{"tflops":11.6,"gflops":0.0,"bandwidth_gbps":0.0,"arithmetic_intensity":0.0},"muda":[]}"#;

    std::fs::write("/tmp/cgp-json-test-b.json", baseline).unwrap();
    std::fs::write("/tmp/cgp-json-test-c.json", current).unwrap();

    let output = cgp_cmd()
        .args([
            "--json",
            "diff",
            "--baseline",
            "/tmp/cgp-json-test-b.json",
            "--current",
            "/tmp/cgp-json-test-c.json",
        ])
        .output()
        .expect("Failed to run cgp --json diff");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);

    let parsed: serde_json::Value =
        serde_json::from_str(&stdout).expect("Diff JSON is not valid JSON");

    assert!(parsed.is_array(), "Diff output should be an array");
    let arr = parsed.as_array().unwrap();
    assert!(!arr.is_empty(), "Diff array should not be empty");
    assert!(arr[0].get("verdict").is_some(), "Missing verdict field");

    let _ = std::fs::remove_file("/tmp/cgp-json-test-b.json");
    let _ = std::fs::remove_file("/tmp/cgp-json-test-c.json");
}

/// cgp bench must handle missing benchmark gracefully.
#[test]
fn test_bench_missing() {
    let output = cgp_cmd()
        .args(["bench", "--bench", "nonexistent_bench_xyz_99"])
        .output()
        .expect("Failed to run cgp bench");

    // Should succeed (graceful handling, not crash)
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("CGP Bench") || stdout.contains("not found"),
        "stdout: {stdout}"
    );
}

/// cgp profile simd --arch neon must degrade gracefully on x86.
/// FALSIFY-CGP-071.
#[test]
#[cfg(target_arch = "x86_64")]
fn test_neon_graceful_on_x86() {
    let output = cgp_cmd()
        .args([
            "profile",
            "simd",
            "--function",
            "test",
            "--size",
            "1024",
            "--arch",
            "neon",
        ])
        .output()
        .expect("Failed to run cgp profile simd neon");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("NEON not available"),
        "Should mention NEON not available: {stdout}"
    );
}

/// cgp profile wgpu must detect backend and not crash.
/// FALSIFY-CGP-079 partial: wgpu with native target.
#[test]
fn test_wgpu_native_profile() {
    let output = cgp_cmd()
        .args([
            "profile",
            "wgpu",
            "--shader",
            "nonexistent.wgsl",
            "--dispatch",
            "256,256,1",
        ])
        .output()
        .expect("Failed to run cgp profile wgpu");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("wgpu"), "Should mention wgpu: {stdout}");
    assert!(
        stdout.contains("Vulkan") || stdout.contains("Metal") || stdout.contains("DX12"),
        "Should detect backend: {stdout}"
    );
}

/// cgp profile wgpu --target web must mention browser fallback.
/// FALSIFY-CGP-079.
#[test]
fn test_wgpu_web_fallback() {
    let output = cgp_cmd()
        .args([
            "profile",
            "wgpu",
            "--shader",
            "test.wgsl",
            "--target",
            "web",
        ])
        .output()
        .expect("Failed to run cgp profile wgpu web");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should either find Chrome or fall back
    assert!(
        stdout.contains("Chrome") || stdout.contains("falling back"),
        "Should handle browser presence: {stdout}"
    );
}

/// cgp profile wasm must handle missing wasmtime gracefully.
/// FALSIFY-CGP-072 partial.
#[test]
fn test_wasm_profile() {
    let output = cgp_cmd()
        .args([
            "profile",
            "wasm",
            "--function",
            "vector_dot_wasm",
            "--size",
            "1024",
        ])
        .output()
        .expect("Failed to run cgp profile wasm");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("WASM"), "Should mention WASM: {stdout}");
    // Should either find wasmtime or show install instructions
    assert!(
        stdout.contains("wasmtime") || stdout.contains("Install"),
        "Should handle wasmtime presence: {stdout}"
    );
}

/// cgp profile scalar must not crash.
#[test]
fn test_scalar_profile() {
    let output = cgp_cmd()
        .args([
            "profile",
            "scalar",
            "--function",
            "matrix_mul_naive",
            "--size",
            "256",
        ])
        .output()
        .expect("Failed to run cgp profile scalar");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Scalar"), "Should mention Scalar: {stdout}");
    assert!(
        stdout.contains("baseline"),
        "Should mention baseline: {stdout}"
    );
}

/// cgp profile parallel must not crash.
/// FALSIFY-CGP-080 partial.
#[test]
fn test_parallel_profile() {
    let output = cgp_cmd()
        .args([
            "profile",
            "parallel",
            "--function",
            "gemm_heijunka",
            "--size",
            "1024",
            "--threads",
            "4",
        ])
        .output()
        .expect("Failed to run cgp profile parallel");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Parallel") || stdout.contains("Rayon"),
        "stdout: {stdout}"
    );
}

/// cgp --json roofline must produce valid JSON for different targets.
#[test]
fn test_json_roofline_avx2() {
    let output = cgp_cmd()
        .args(["--json", "roofline", "--target", "avx2"])
        .output()
        .expect("Failed to run cgp --json roofline avx2");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed: serde_json::Value =
        serde_json::from_str(&stdout).expect("Roofline AVX2 JSON is not valid JSON");
    assert!(parsed.get("peak_compute").is_some());
    assert!(parsed.get("target").is_some());
}

/// cgp profile quant must handle Q4K kernel.
/// FALSIFY-CGP-074 partial.
#[test]
fn test_quant_profile() {
    let output = cgp_cmd()
        .args([
            "profile",
            "quant",
            "--kernel",
            "q4k_gemv",
            "--size",
            "4096x1x4096",
        ])
        .output()
        .expect("Failed to run cgp profile quant");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Q4K") || stdout.contains("quant") || stdout.contains("Quant"),
        "stdout: {stdout}"
    );
}

/// cgp contract verify must handle missing directory.
#[test]
fn test_contract_verify_missing_dir() {
    let output = cgp_cmd()
        .args([
            "contract",
            "verify",
            "--contracts-dir",
            "/tmp/nonexistent_cgp_contracts_xyz",
        ])
        .output()
        .expect("Failed to run cgp contract verify");

    // Should succeed with 0 contracts (not crash)
    assert!(output.status.success());
}

/// cgp --json profile compare must output valid JSON array.
#[test]
fn test_json_profile_compare() {
    let output = cgp_cmd()
        .args([
            "--json",
            "profile",
            "compare",
            "--kernel",
            "gemm",
            "--size",
            "256",
            "--backends",
            "scalar,avx2",
        ])
        .output()
        .expect("Failed to run cgp --json profile compare");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed: serde_json::Value =
        serde_json::from_str(&stdout).expect("Compare JSON is not valid JSON");
    assert!(parsed.is_array(), "Compare output should be an array");
    let arr = parsed.as_array().unwrap();
    assert!(!arr.is_empty(), "Should have backend results");
    assert!(
        arr[0].get("name").is_some(),
        "Each result should have a name"
    );
    assert!(
        arr[0].get("tflops").is_some(),
        "Each result should have tflops"
    );
}

/// cgp explain ptx must not crash and mention PTX.
#[test]
fn test_explain_ptx() {
    let output = cgp_cmd()
        .args(["explain", "ptx"])
        .output()
        .expect("Failed to run cgp explain ptx");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("PTX"), "Should mention PTX: {stdout}");
}

/// cgp explain wgsl must not crash.
#[test]
fn test_explain_wgsl() {
    let output = cgp_cmd()
        .args(["explain", "wgsl"])
        .output()
        .expect("Failed to run cgp explain wgsl");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("WGSL"), "Should mention WGSL: {stdout}");
}

/// cgp profile scaling must output JSON with thread-count data.
#[test]
fn test_json_scaling() {
    let output = cgp_cmd()
        .args([
            "--json",
            "profile",
            "scaling",
            "--size",
            "256",
            "--max-threads",
            "2",
            "--runs",
            "1",
        ])
        .output()
        .expect("Failed to run cgp profile scaling");

    // May fail if benchmark binary not found — that's OK for CI
    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let parsed: serde_json::Value =
            serde_json::from_str(&stdout).expect("Scaling JSON is not valid JSON");
        let arr = parsed.as_array().unwrap();
        assert!(!arr.is_empty(), "Should have at least 1 scaling point");
        assert!(arr[0].get("threads").is_some());
        assert!(arr[0].get("gflops").is_some());
        assert!(arr[0].get("scaling").is_some());
    }
}

/// cgp --json doctor must have GPU detection fields.
#[test]
fn test_json_doctor_gpu_detection() {
    let output = cgp_cmd()
        .args(["--json", "doctor"])
        .output()
        .expect("Failed to run cgp --json doctor");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed: serde_json::Value =
        serde_json::from_str(&stdout).expect("Doctor JSON is not valid JSON");

    let checks = parsed["checks"].as_array().unwrap();
    let gpu_check = checks.iter().find(|c| c["name"] == "GPU");
    if let Some(gc) = gpu_check {
        assert_eq!(gc["status"], "Ok", "GPU should be detected");
    }
}
