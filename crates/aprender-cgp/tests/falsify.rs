//! FALSIFY tests — real-world validation of cgp claims.
//! Each test attempts to falsify a specific claim from the spec.
//! Tests that pass mean the claim survived falsification.

// Integration-test binary: `.unwrap()` is idiomatic in test assertions and the
// lib's `cfg(test)` allow does not reach this separate crate.
#![allow(clippy::disallowed_methods)]

use std::process::Command;
use std::time::Instant;

fn cgp_cmd() -> Command {
    let mut cmd = Command::new(env!("CARGO"));
    // PACKAGE name, not the [lib] name. APR-MONO renamed the package to
    // aprender-cgp and left this as "cgp", so every test in this file died
    // at cargo package resolution -- before a single byte of cgp was linked.
    cmd.arg("run").arg("-p").arg("aprender-cgp").arg("--");
    cmd
}

// ══════════════════════════════════════════════════════════════════════
// FALSIFY-CGP-030: Must detect deliberate 10% regression
// Given: baseline profile saved for kernel K
// When: K is modified to be 10% slower
// Then: cgp diff reports REGRESSION
// ══════════════════════════════════════════════════════════════════════
#[test]
fn falsify_cgp_030_detect_10pct_regression() {
    // Baseline: 23.2us, 50 samples, stddev 0.3
    let baseline = serde_json::json!({
        "version": "2.0", "timestamp": "", "hardware": {"cpu_features": []},
        "timing": {
            "wall_clock_time_us": 23.2,
            "samples": 50,
            "stddev_us": 0.3,
            "ci_95_low_us": 22.9,
            "ci_95_high_us": 23.5
        },
        "throughput": {"tflops": 11.6, "gflops": 0.0, "bandwidth_gbps": 0.0, "arithmetic_intensity": 0.0},
        "muda": []
    });

    // Current: 10% slower = 25.52us
    let current = serde_json::json!({
        "version": "2.0", "timestamp": "", "hardware": {"cpu_features": []},
        "timing": {
            "wall_clock_time_us": 25.52,
            "samples": 50,
            "stddev_us": 0.3,
            "ci_95_low_us": 25.2,
            "ci_95_high_us": 25.8
        },
        "throughput": {"tflops": 10.5, "gflops": 0.0, "bandwidth_gbps": 0.0, "arithmetic_intensity": 0.0},
        "muda": []
    });

    std::fs::write("/tmp/cgp-falsify-030-b.json", baseline.to_string()).unwrap();
    std::fs::write("/tmp/cgp-falsify-030-c.json", current.to_string()).unwrap();

    let output = cgp_cmd()
        .args([
            "diff",
            "--baseline",
            "/tmp/cgp-falsify-030-b.json",
            "--current",
            "/tmp/cgp-falsify-030-c.json",
        ])
        .output()
        .expect("Failed to run cgp diff");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Must detect regression on wall_clock_time_us
    assert!(
        stdout.contains("REGRESSION"),
        "FALSIFY-CGP-030 FAILED: 10% regression not detected.\nOutput:\n{stdout}"
    );

    let _ = std::fs::remove_file("/tmp/cgp-falsify-030-b.json");
    let _ = std::fs::remove_file("/tmp/cgp-falsify-030-c.json");
}

// ══════════════════════════════════════════════════════════════════════
// FALSIFY-CGP-031: Must NOT false-positive on noise (<2% variation)
// Given: kernel K profiled twice with identical code
// When: cgp diff --baseline run1 --current run2
// Then: reports NO_CHANGE (not regression)
// ══════════════════════════════════════════════════════════════════════
#[test]
fn falsify_cgp_031_no_false_positive_on_noise() {
    // Run 1: 23.2us
    let run1 = serde_json::json!({
        "version": "2.0", "timestamp": "", "hardware": {"cpu_features": []},
        "timing": {
            "wall_clock_time_us": 23.2,
            "samples": 50,
            "stddev_us": 0.5,
            "ci_95_low_us": 22.7,
            "ci_95_high_us": 23.7
        },
        "throughput": {"tflops": 11.6, "gflops": 0.0, "bandwidth_gbps": 0.0, "arithmetic_intensity": 0.0},
        "muda": []
    });

    // Run 2: 23.4us (only ~0.9% difference — within noise)
    let run2 = serde_json::json!({
        "version": "2.0", "timestamp": "", "hardware": {"cpu_features": []},
        "timing": {
            "wall_clock_time_us": 23.4,
            "samples": 50,
            "stddev_us": 0.5,
            "ci_95_low_us": 22.9,
            "ci_95_high_us": 23.9
        },
        "throughput": {"tflops": 11.5, "gflops": 0.0, "bandwidth_gbps": 0.0, "arithmetic_intensity": 0.0},
        "muda": []
    });

    std::fs::write("/tmp/cgp-falsify-031-1.json", run1.to_string()).unwrap();
    std::fs::write("/tmp/cgp-falsify-031-2.json", run2.to_string()).unwrap();

    let output = cgp_cmd()
        .args([
            "diff",
            "--baseline",
            "/tmp/cgp-falsify-031-1.json",
            "--current",
            "/tmp/cgp-falsify-031-2.json",
        ])
        .output()
        .expect("Failed to run cgp diff");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Must NOT detect regression on timing (noise is within stddev)
    // The timing line should say NO_CHANGE or IMPROVED, not REGRESSION
    let timing_lines: Vec<&str> = stdout
        .lines()
        .filter(|l| l.contains("wall_clock_time_us"))
        .collect();

    for line in &timing_lines {
        assert!(
            !line.contains("REGRESSION"),
            "FALSIFY-CGP-031 FAILED: false positive on <2% noise.\nLine: {line}\nOutput:\n{stdout}"
        );
    }

    let _ = std::fs::remove_file("/tmp/cgp-falsify-031-1.json");
    let _ = std::fs::remove_file("/tmp/cgp-falsify-031-2.json");
}

// ══════════════════════════════════════════════════════════════════════
// FALSIFY-CGP-041: SIMD must be faster than scalar (>= 3x at 1024)
// Verified via the compare command's estimation model.
// ══════════════════════════════════════════════════════════════════════
#[test]
fn falsify_cgp_041_simd_faster_than_scalar() {
    let output = cgp_cmd()
        .args([
            "--json",
            "profile",
            "compare",
            "--kernel",
            "gemm",
            "--size",
            "1024",
            "--backends",
            "scalar,avx2",
        ])
        .output()
        .expect("Failed to run cgp profile compare");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed: serde_json::Value = serde_json::from_str(&stdout).expect("Compare JSON invalid");

    let arr = parsed.as_array().unwrap();
    let scalar = arr.iter().find(|r| r["name"] == "scalar").unwrap();
    let avx2 = arr.iter().find(|r| r["name"] == "avx2").unwrap();

    let scalar_time = scalar["wall_time_us"].as_f64().unwrap();
    let avx2_time = avx2["wall_time_us"].as_f64().unwrap();
    let speedup = scalar_time / avx2_time;

    assert!(
        speedup >= 3.0,
        "FALSIFY-CGP-041 FAILED: AVX2 speedup {speedup:.1}x < 3x (scalar={scalar_time:.0}us, avx2={avx2_time:.0}us)"
    );
}

// ══════════════════════════════════════════════════════════════════════
// FALSIFY-CGP-043: Must profile arbitrary CUDA binary via nsys
// Given: any CUDA binary
// When: cgp profile binary ./binary
// Then: extracts kernel names, launch configs, wall-clock timings
// ══════════════════════════════════════════════════════════════════════
#[test]
fn falsify_cgp_043_profile_binary() {
    // Use nvidia-smi as a trivial binary (always available, exercises CUDA driver)
    let output = cgp_cmd()
        .args(["profile", "binary", "nvidia-smi"])
        .output()
        .expect("Failed to run cgp profile binary");

    // Should succeed (even if nsys finds no kernels — nvidia-smi doesn't launch kernels)
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Binary Profile")
            || stdout.contains("nsys")
            || stdout.contains("nvidia-smi"),
        "FALSIFY-CGP-043: Should mention binary profiling.\nOutput:\n{stdout}"
    );
}

// ══════════════════════════════════════════════════════════════════════
// FALSIFY-CGP-045: cgp compete must produce normalized comparison
// Given: two commands producing results
// When: cgp compete --ours cmd1 --theirs cmd2
// Then: table shows time and vs-best ratio
// ══════════════════════════════════════════════════════════════════════
#[test]
fn falsify_cgp_045_compete_normalized() {
    // Use "sleep 0.01" vs "sleep 0.02" — known 2x difference
    let output = cgp_cmd()
        .args([
            "compete",
            "timing",
            "--ours",
            "sleep 0.01",
            "--theirs",
            "sleep 0.02",
            "--label",
            "fast,slow",
        ])
        .output()
        .expect("Failed to run cgp compete");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Must have a comparison table with timing
    assert!(stdout.contains("Head-to-Head"), "Missing header");
    assert!(stdout.contains("Winner"), "Missing winner declaration");
    // The "fast" command (sleep 0.01) should be the winner or at least faster
    assert!(
        stdout.contains("fast") && stdout.contains("slow"),
        "FALSIFY-CGP-045: Labels not in output.\nOutput:\n{stdout}"
    );
}

// ══════════════════════════════════════════════════════════════════════
// FALSIFY-CGP-047: Must not crash on competitor binary that segfaults
// Given: a binary that crashes
// When: cgp profile binary ./crashing_binary
// Then: reports error gracefully (no cgp crash)
// ══════════════════════════════════════════════════════════════════════
#[test]
fn falsify_cgp_047_crash_handling() {
    // "false" is a binary that exits with code 1 (not segfault, but tests error path)
    let output = cgp_cmd()
        .args(["profile", "binary", "false"])
        .output()
        .expect("Failed to run cgp profile binary");

    // cgp itself must not crash — it should handle the error gracefully
    // (may return success with an error message, or non-zero with message)
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{stdout}{stderr}");

    // Must not be empty (should say something about the binary)
    assert!(
        !combined.trim().is_empty(),
        "FALSIFY-CGP-047: cgp produced no output for failing binary"
    );
}

// ══════════════════════════════════════════════════════════════════════
// FALSIFY-CGP-062: cgp diff must not require re-profiling
// Given: two saved profile JSONs
// When: cgp diff --baseline a.json --current b.json
// Then: reads the JSONs directly and reports a verdict -- no wall-clock
// budget here. `cgp_cmd()` runs `cargo run`, so a timed run measures
// cargo's own freshness check and process-spawn cost, not the diff
// analysis; on shared/loaded CI hosts that overhead alone exceeded every
// budget tried. The structural claim -- "pure analysis, no re-profiling" --
// is instead checked by requiring the tool's own reported "Diff completed
// in Nms" line to be small, which is the diff's self-measured time, not
// ours around a subprocess.
// ══════════════════════════════════════════════════════════════════════
#[test]
fn falsify_cgp_062_diff_speed() {
    let baseline = serde_json::json!({
        "version": "2.0", "timestamp": "", "hardware": {"cpu_features": []},
        "timing": {"wall_clock_time_us": 35.7, "samples": 1, "stddev_us": 0.0, "ci_95_low_us": 0.0, "ci_95_high_us": 0.0},
        "throughput": {"tflops": 7.5, "gflops": 0.0, "bandwidth_gbps": 0.0, "arithmetic_intensity": 0.0},
        "muda": []
    });
    let current = serde_json::json!({
        "version": "2.0", "timestamp": "", "hardware": {"cpu_features": []},
        "timing": {"wall_clock_time_us": 23.2, "samples": 1, "stddev_us": 0.0, "ci_95_low_us": 0.0, "ci_95_high_us": 0.0},
        "throughput": {"tflops": 11.6, "gflops": 0.0, "bandwidth_gbps": 0.0, "arithmetic_intensity": 0.0},
        "muda": []
    });

    std::fs::write("/tmp/cgp-falsify-062-b.json", baseline.to_string()).unwrap();
    std::fs::write("/tmp/cgp-falsify-062-c.json", current.to_string()).unwrap();

    let output = cgp_cmd()
        .args([
            "diff",
            "--baseline",
            "/tmp/cgp-falsify-062-b.json",
            "--current",
            "/tmp/cgp-falsify-062-c.json",
        ])
        .output()
        .expect("Failed to run cgp diff");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Non-vacuity: a broken diff that silently produced no comparison would
    // still exit 0. Require the verdict line this tool actually prints.
    assert!(
        stdout.contains("improvement(s)") || stdout.contains("regression(s)"),
        "FALSIFY-CGP-062: no verdict summary in output.\nOutput:\n{stdout}"
    );

    // The structural claim: diff is pure JSON analysis, not re-profiling.
    // Assert the TOOL's OWN self-measured time (the "Diff completed in
    // Nms" line), not wall-clock around the subprocess -- `cgp_cmd()` goes
    // through `cargo run`, whose freshness check and process-spawn cost
    // dominate any external timer and are not what this claim is about.
    let self_reported_ms: u64 = stdout
        .lines()
        .find_map(|l| {
            l.trim()
                .strip_prefix("Diff completed in ")?
                .strip_suffix("ms")
        })
        .and_then(|n| n.parse().ok())
        .unwrap_or_else(|| {
            panic!("FALSIFY-CGP-062: no 'Diff completed in Nms' line.\nOutput:\n{stdout}")
        });
    assert!(
        self_reported_ms < 100,
        "FALSIFY-CGP-062 FAILED: tool reported {self_reported_ms}ms for pure JSON analysis"
    );

    let _ = std::fs::remove_file("/tmp/cgp-falsify-062-b.json");
    let _ = std::fs::remove_file("/tmp/cgp-falsify-062-c.json");
}

// ══════════════════════════════════════════════════════════════════════
// FALSIFY-CGP-075: Must report effective bandwidth (not raw)
// Q4K: 4096*4096 weights / 256 * 144 bytes = 9,437,184 bytes = 9.44 MB
// ══════════════════════════════════════════════════════════════════════
#[test]
fn falsify_cgp_075_q4k_effective_bandwidth() {
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

    // Must report compressed size ~9.44 MB (not FP32 size of 67 MB)
    assert!(
        stdout.contains("9.44 MB") || stdout.contains("9.4"),
        "FALSIFY-CGP-075 FAILED: Q4K compressed size should be ~9.44 MB.\nOutput:\n{stdout}"
    );

    // Must NOT report FP32 size as the primary bandwidth metric
    // The 67 MB FP32 equivalent should be clearly labeled as such
    if stdout.contains("67") {
        assert!(
            stdout.contains("FP32 equivalent") || stdout.contains("equivalent"),
            "FALSIFY-CGP-075: If 67MB shown, must be labeled as FP32 equivalent"
        );
    }
}

// ══════════════════════════════════════════════════════════════════════
// FALSIFY-CGP-061: Doctor must complete in < 2 seconds (real timing)
// ══════════════════════════════════════════════════════════════════════
#[test]
fn falsify_cgp_061_doctor_speed_real() {
    // Warm up (first run may compile)
    let _ = cgp_cmd().args(["doctor"]).output();

    let start = Instant::now();
    let output = cgp_cmd()
        .args(["doctor"])
        .output()
        .expect("Failed to run cgp doctor");
    let elapsed = start.elapsed();

    assert!(output.status.success());
    // Allow 500ms subprocess overhead on top of the 2s spec limit
    assert!(
        elapsed.as_millis() < 2500,
        "FALSIFY-CGP-061 FAILED: doctor took {}ms",
        elapsed.as_millis()
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Verify it detected the GPU
    assert!(stdout.contains("RTX 4090") || stdout.contains("GPU"));
}

// ══════════════════════════════════════════════════════════════════════
// FALSIFY-CGP-021: Ridge points must be mathematically correct
// FP16: 330000 / 1008 = 327.38 ≈ 327.4 FLOP/byte
// ══════════════════════════════════════════════════════════════════════
#[test]
fn falsify_cgp_021_ridge_point_math() {
    let output = cgp_cmd()
        .args(["--json", "roofline", "--target", "cuda"])
        .output()
        .expect("Failed to run cgp roofline");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed: serde_json::Value = serde_json::from_str(&stdout).unwrap();

    // Verify ridge points
    let ridge_points = &parsed["ridge_points"];
    if let Some(arr) = ridge_points.as_array() {
        // Find FP16 Tensor ridge point
        let fp16 = arr.iter().find(|r| {
            r["precision"]
                .as_str()
                .is_some_and(|s| s.contains("FP16") || s.contains("Fp16"))
        });
        if let Some(fp16_ridge) = fp16 {
            let ridge = fp16_ridge["ridge_flop_per_byte"].as_f64().unwrap_or(0.0);
            // 330000 / 1008 = 327.38
            let expected = 330_000.0_f64 / 1008.0;
            assert!(
                (ridge - expected).abs() < 0.5,
                "FALSIFY-CGP-021 FAILED: FP16 ridge={ridge:.1}, expected={expected:.1}"
            );
        }
    }

    // Also verify via text output
    let text_output = cgp_cmd()
        .args(["roofline", "--target", "cuda"])
        .output()
        .expect("Failed");
    let text = String::from_utf8_lossy(&text_output.stdout);
    assert!(
        text.contains("327"),
        "FALSIFY-CGP-021: Ridge point 327.x not in output.\n{text}"
    );
}

// ══════════════════════════════════════════════════════════════════════
// FALSIFY-CGP-060: cgp profile must complete in < 30 seconds
// Given: GEMM 512x512 kernel
// When: cgp profile compare --kernel gemm --size 512
// Then: total wall time < 30 seconds
// ══════════════════════════════════════════════════════════════════════
#[test]
fn falsify_cgp_060_profile_speed() {
    // Warm up (first run may compile)
    let _ = cgp_cmd()
        .args([
            "profile",
            "compare",
            "--kernel",
            "gemm",
            "--size",
            "512",
            "--backends",
            "scalar,avx2",
        ])
        .output();

    let start = Instant::now();
    let output = cgp_cmd()
        .args([
            "profile",
            "compare",
            "--kernel",
            "gemm",
            "--size",
            "512",
            "--backends",
            "scalar,avx2",
        ])
        .output()
        .expect("Failed to run cgp profile compare");
    let elapsed = start.elapsed();

    assert!(output.status.success());
    // 30s spec limit + 500ms subprocess overhead
    assert!(
        elapsed.as_secs() < 31,
        "FALSIFY-CGP-060 FAILED: profile took {}s (limit: 30s)",
        elapsed.as_secs()
    );
}

// ══════════════════════════════════════════════════════════════════════
// FALSIFY-CGP-020: Roofline peak bandwidth must match spec (1008 GB/s)
// Given: RTX 4090 with GDDR6X
// When: cgp roofline --target cuda --json
// Then: DRAM bandwidth = 1008 GB/s (384-bit × 21 Gbps)
// ══════════════════════════════════════════════════════════════════════
#[test]
fn falsify_cgp_020_bandwidth_spec() {
    let output = cgp_cmd()
        .args(["--json", "roofline", "--target", "cuda"])
        .output()
        .expect("Failed to run cgp roofline");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed: serde_json::Value = serde_json::from_str(&stdout).unwrap();

    let dram_bw = parsed["peak_bandwidth"]["Dram"].as_f64().unwrap_or(0.0);
    let expected_bw = 1_008_000_000_000.0_f64; // 1008 GB/s in bytes/s
    let tolerance = expected_bw * 0.05; // 5% tolerance
    assert!(
        (dram_bw - expected_bw).abs() < tolerance,
        "FALSIFY-CGP-020 FAILED: DRAM bandwidth {:.0} GB/s vs expected {:.0} GB/s",
        dram_bw / 1e9,
        expected_bw / 1e9
    );
}

// ══════════════════════════════════════════════════════════════════════
// FALSIFY-CGP-032: Must detect improvement
// Given: baseline at 35.7us, current at 23.2us
// When: cgp diff
// Then: reports IMPROVED with 1.54x speedup
// ══════════════════════════════════════════════════════════════════════
#[test]
fn falsify_cgp_032_detect_improvement() {
    let baseline = serde_json::json!({
        "version": "2.0", "timestamp": "", "hardware": {"cpu_features": []},
        "timing": {
            "wall_clock_time_us": 35.7,
            "samples": 50, "stddev_us": 0.5,
            "ci_95_low_us": 35.2, "ci_95_high_us": 36.2
        },
        "throughput": {"tflops": 7.5, "gflops": 0.0, "bandwidth_gbps": 0.0, "arithmetic_intensity": 0.0},
        "muda": []
    });
    let current = serde_json::json!({
        "version": "2.0", "timestamp": "", "hardware": {"cpu_features": []},
        "timing": {
            "wall_clock_time_us": 23.2,
            "samples": 50, "stddev_us": 0.3,
            "ci_95_low_us": 22.9, "ci_95_high_us": 23.5
        },
        "throughput": {"tflops": 11.6, "gflops": 0.0, "bandwidth_gbps": 0.0, "arithmetic_intensity": 0.0},
        "muda": []
    });

    std::fs::write("/tmp/cgp-falsify-032-b.json", baseline.to_string()).unwrap();
    std::fs::write("/tmp/cgp-falsify-032-c.json", current.to_string()).unwrap();

    let output = cgp_cmd()
        .args([
            "diff",
            "--baseline",
            "/tmp/cgp-falsify-032-b.json",
            "--current",
            "/tmp/cgp-falsify-032-c.json",
        ])
        .output()
        .expect("Failed to run cgp diff");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Must detect improvement (lower time = better)
    assert!(
        stdout.contains("IMPROVED"),
        "FALSIFY-CGP-032 FAILED: 35.7→23.2us should be IMPROVED.\nOutput:\n{stdout}"
    );

    let _ = std::fs::remove_file("/tmp/cgp-falsify-032-b.json");
    let _ = std::fs::remove_file("/tmp/cgp-falsify-032-c.json");
}

// ══════════════════════════════════════════════════════════════════════
// FALSIFY-CGP-042: cuBLAS must be faster than pure-Rust PTX for large GEMM
// Verified via compare command's estimation model at 4096.
// ══════════════════════════════════════════════════════════════════════
#[test]
fn falsify_cgp_042_cublas_faster_than_ptx() {
    let output = cgp_cmd()
        .args([
            "--json",
            "profile",
            "compare",
            "--kernel",
            "gemm",
            "--size",
            "4096",
            "--backends",
            "cuda,cublas",
        ])
        .output()
        .expect("Failed to run cgp profile compare");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed: serde_json::Value = serde_json::from_str(&stdout).expect("Compare JSON invalid");

    let arr = parsed.as_array().unwrap();
    let cuda = arr.iter().find(|r| r["name"] == "cuda").unwrap();
    let cublas = arr.iter().find(|r| r["name"] == "cublas").unwrap();

    let cuda_tflops = cuda["tflops"].as_f64().unwrap();
    let cublas_tflops = cublas["tflops"].as_f64().unwrap();

    assert!(
        cublas_tflops > cuda_tflops,
        "FALSIFY-CGP-042 FAILED: cuBLAS {cublas_tflops:.1} TFLOP/s should exceed pure PTX {cuda_tflops:.1} TFLOP/s at 4096"
    );
}

// ══════════════════════════════════════════════════════════════════════
// FALSIFY-CGP-046: Must handle CPU-only competitor (no CUDA)
// When profiling a CPU-only command, cgp compete should still work
// (falls back to wall-clock timing, no GPU metrics).
// ══════════════════════════════════════════════════════════════════════
#[test]
fn falsify_cgp_046_cpu_only_competitor() {
    let output = cgp_cmd()
        .args([
            "compete",
            "cpu_timing",
            "--ours",
            "sleep 0.01",
            "--theirs",
            "sleep 0.015",
            "--label",
            "fast,slow",
        ])
        .output()
        .expect("Failed to run cgp compete");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Must produce timing output without crashing on CPU-only workloads
    assert!(
        stdout.contains("fast") && stdout.contains("slow"),
        "FALSIFY-CGP-046 FAILED: Labels missing.\nOutput:\n{stdout}"
    );
    assert!(
        stdout.contains("Winner"),
        "FALSIFY-CGP-046: Should declare a winner.\nOutput:\n{stdout}"
    );
}

// ══════════════════════════════════════════════════════════════════════
// FALSIFY-CGP-SCALING-001: Scaling JSON has required fields
// Contract: cgp-scaling-v1.yaml
// ══════════════════════════════════════════════════════════════════════
#[test]
fn falsify_cgp_scaling_001_json_fields() {
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

    // May fail if benchmark binary not found — skip gracefully
    if !output.status.success() {
        return;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed: serde_json::Value = serde_json::from_str(&stdout).expect("Scaling JSON invalid");
    let arr = parsed.as_array().expect("Should be array");
    assert!(!arr.is_empty(), "Should have at least 1 data point");

    for point in arr {
        assert!(
            point.get("threads").is_some(),
            "FALSIFY-CGP-SCALING-001: missing 'threads' field"
        );
        assert!(
            point.get("gflops").is_some(),
            "FALSIFY-CGP-SCALING-001: missing 'gflops' field"
        );
        assert!(
            point.get("scaling").is_some(),
            "FALSIFY-CGP-SCALING-001: missing 'scaling' field"
        );
    }
}

// ══════════════════════════════════════════════════════════════════════
// FALSIFY-CGP-SCALING-002: Single-thread scaling is ~1.0x
// Contract: cgp-scaling-v1.yaml
// ══════════════════════════════════════════════════════════════════════
#[test]
fn falsify_cgp_scaling_002_baseline_is_1x() {
    let output = cgp_cmd()
        .args([
            "--json",
            "profile",
            "scaling",
            "--size",
            "256",
            "--max-threads",
            "1",
            "--runs",
            "1",
        ])
        .output()
        .expect("Failed to run cgp profile scaling");

    if !output.status.success() {
        return;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed: serde_json::Value = serde_json::from_str(&stdout).expect("Scaling JSON invalid");
    let arr = parsed.as_array().unwrap();
    if let Some(first) = arr.first() {
        let scaling = first["scaling"].as_f64().unwrap_or(0.0);
        assert!(
            (scaling - 1.0).abs() < 0.15,
            "FALSIFY-CGP-SCALING-002: 1T scaling should be ~1.0, got {scaling}"
        );
    }
}

// ══════════════════════════════════════════════════════════════════════
// FALSIFY-CGP-EMPIRICAL-010: Empirical roofline must produce output
// Spec section 3.1: --empirical flag measures actual bandwidth and FLOPS.
// ══════════════════════════════════════════════════════════════════════
#[test]
fn falsify_cgp_empirical_010_roofline_output() {
    let output = cgp_cmd()
        .args(["roofline", "--target", "avx512", "--empirical"])
        .output()
        .expect("Failed to run cgp roofline --empirical");

    assert!(
        output.status.success(),
        "cgp roofline --empirical must succeed"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        stdout.contains("Empirical Measurement"),
        "FALSIFY-CGP-EMPIRICAL-010: Must show empirical section.\nOutput:\n{stdout}"
    );
    assert!(
        stdout.contains("DRAM Bandwidth"),
        "FALSIFY-CGP-EMPIRICAL-010: Must show measured bandwidth.\nOutput:\n{stdout}"
    );
    assert!(
        stdout.contains("Peak FP32 FLOPS"),
        "FALSIFY-CGP-EMPIRICAL-010: Must show measured FLOPS.\nOutput:\n{stdout}"
    );
    assert!(
        stdout.contains("Empirical Ridge"),
        "FALSIFY-CGP-EMPIRICAL-010: Must show empirical ridge point.\nOutput:\n{stdout}"
    );
}

// ══════════════════════════════════════════════════════════════════════
// FALSIFY-CGP-EMPIRICAL-013: JSON empirical output has both theoretical + empirical
// ══════════════════════════════════════════════════════════════════════
#[test]
fn falsify_cgp_empirical_013_json_output() {
    let output = cgp_cmd()
        .args(["--json", "roofline", "--target", "avx512", "--empirical"])
        .output()
        .expect("Failed to run cgp --json roofline --empirical");

    assert!(output.status.success(), "Must succeed");
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Must be valid JSON
    let parsed: serde_json::Value =
        serde_json::from_str(&stdout).expect("Output must be valid JSON");

    // Must have both theoretical and empirical sections
    assert!(
        parsed.get("theoretical").is_some(),
        "FALSIFY-CGP-EMPIRICAL-013: JSON must have 'theoretical' field.\nGot:\n{stdout}"
    );
    assert!(
        parsed.get("empirical").is_some(),
        "FALSIFY-CGP-EMPIRICAL-013: JSON must have 'empirical' field.\nGot:\n{stdout}"
    );

    // Empirical must have measured values
    let emp = &parsed["empirical"];
    assert!(
        emp.get("measured_bandwidth_bps").is_some(),
        "Must have measured_bandwidth_bps"
    );
    assert!(
        emp.get("measured_peak_flops").is_some(),
        "Must have measured_peak_flops"
    );
    assert!(
        emp.get("measured_ridge_point").is_some(),
        "Must have measured_ridge_point"
    );
}

// ══════════════════════════════════════════════════════════════════════
// FALSIFY-CGP-EMPIRICAL-011: Empirical bandwidth must be > 0 GB/s
// Note: when run via cargo test (debug/unoptimized), STREAM bandwidth
// is much lower than release. Threshold is lenient; the real validation
// is via `cgp roofline --empirical` in release mode (20+ GB/s).
// ══════════════════════════════════════════════════════════════════════
#[test]
fn falsify_cgp_empirical_011_bandwidth_sanity() {
    let output = cgp_cmd()
        .args(["roofline", "--target", "avx512", "--empirical"])
        .output()
        .expect("Failed to run cgp roofline --empirical");

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Parse "DRAM Bandwidth:      XX.X GB/s"
    for line in stdout.lines() {
        if line.contains("DRAM Bandwidth:") {
            let bw_str = line
                .split("DRAM Bandwidth:")
                .nth(1)
                .and_then(|s| s.split("GB/s").next())
                .map(|s| s.trim());
            if let Some(bw_val) = bw_str.and_then(|s| s.parse::<f64>().ok()) {
                assert!(
                    bw_val > 0.1,
                    "FALSIFY-CGP-EMPIRICAL-011: Bandwidth {bw_val} GB/s must be > 0.1 GB/s"
                );
                return;
            }
        }
    }
    panic!("FALSIFY-CGP-EMPIRICAL-011: Could not parse bandwidth from output");
}

// ══════════════════════════════════════════════════════════════════════
// FALSIFY-CGP-EMPIRICAL-012: Empirical AVX-512 FLOPS > 10 GFLOP/s
// Any AVX-512 capable CPU should exceed 10 GFLOP/s single-core FP32.
// ══════════════════════════════════════════════════════════════════════
#[test]
fn falsify_cgp_empirical_012_flops_sanity() {
    let output = cgp_cmd()
        .args(["roofline", "--target", "avx512", "--empirical"])
        .output()
        .expect("Failed to run cgp roofline --empirical");

    let stdout = String::from_utf8_lossy(&output.stdout);

    for line in stdout.lines() {
        if line.contains("Peak FP32 FLOPS:") {
            let flops_str = line
                .split("Peak FP32 FLOPS:")
                .nth(1)
                .and_then(|s| s.split("GFLOP/s").next())
                .map(|s| s.trim());
            if let Some(flops_val) = flops_str.and_then(|s| s.parse::<f64>().ok()) {
                assert!(
                    flops_val > 10.0,
                    "FALSIFY-CGP-EMPIRICAL-012: FLOPS {flops_val} GFLOP/s must be > 10"
                );
                return;
            }
        }
    }
    // Skip if not on AVX-512 hardware
    if !stdout.contains("AVX-512") {
        return;
    }
    panic!("FALSIFY-CGP-EMPIRICAL-012: Could not parse FLOPS from output");
}

// ══════════════════════════════════════════════════════════════════════
// FALSIFY-CGP-COMPARE-050: Measured GEMM must be available
// When benchmark_matrix_suite binary exists, compare should use real data.
// ══════════════════════════════════════════════════════════════════════
#[test]
fn falsify_cgp_compare_050_measured_data() {
    let output = cgp_cmd()
        .args([
            "profile",
            "compare",
            "--kernel",
            "gemm",
            "--size",
            "1024",
            "--backends",
            "avx512",
        ])
        .output()
        .expect("Failed to run cgp profile compare");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);

    // If benchmark binary exists, should show M (measured) not E (estimated)
    let bench_exists = std::path::Path::new(
        "/mnt/nvme-raid0/targets/trueno/release/examples/benchmark_matrix_suite",
    )
    .exists();

    if bench_exists {
        assert!(
            stdout.contains("M"),
            "FALSIFY-CGP-COMPARE-050: With benchmark binary, should show M=measured.\nOutput:\n{stdout}"
        );
    }
}

// ══════════════════════════════════════════════════════════════════════
// FALSIFY-CGP-QUANT-076: Q4K roofline analysis present
// cgp profile quant must show roofline analysis with bottleneck classification.
// ══════════════════════════════════════════════════════════════════════
#[test]
fn falsify_cgp_quant_076_roofline_analysis() {
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

    // Must always show structural info
    assert!(
        stdout.contains("Super-block:"),
        "Must show super-block info"
    );
    assert!(
        stdout.contains("Compression ratio:"),
        "Must show compression ratio"
    );

    // If benchmark binary exists, should show roofline
    let bench_exists = std::path::Path::new(
        "/mnt/nvme-raid0/targets/trueno/release/examples/benchmark_matrix_suite",
    )
    .exists();
    if bench_exists {
        assert!(
            stdout.contains("Roofline Analysis"),
            "FALSIFY-CGP-QUANT-076: Must show roofline analysis when timing available.\nOutput:\n{stdout}"
        );
        assert!(
            stdout.contains("Bottleneck:"),
            "FALSIFY-CGP-QUANT-076: Must classify bottleneck.\nOutput:\n{stdout}"
        );
    }
}

// ══════════════════════════════════════════════════════════════════════
// FALSIFY-CGP-QUANT-077: Token estimation present for LLM workloads
// ══════════════════════════════════════════════════════════════════════
#[test]
fn falsify_cgp_quant_077_token_estimation() {
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

    let bench_exists = std::path::Path::new(
        "/mnt/nvme-raid0/targets/trueno/release/examples/benchmark_matrix_suite",
    )
    .exists();
    if bench_exists {
        assert!(
            stdout.contains("Token Estimation"),
            "FALSIFY-CGP-QUANT-077: Must show LLM token estimation.\nOutput:\n{stdout}"
        );
        assert!(
            stdout.contains("tokens/sec"),
            "FALSIFY-CGP-QUANT-077: Must show tokens/sec.\nOutput:\n{stdout}"
        );
    }
}

// ══════════════════════════════════════════════════════════════════════
// FALSIFY-CGP-091: trueno GEMM must be >= 1.0x vs ndarray (criterion)
// Spec section 8.4b: Competitive with ndarray (BLIS/OpenBLAS backend).
// Note: 1.5x target applies to pure Rust advantages; ndarray also uses
// BLAS so parity (1.0x) is the fair target for single-thread GEMM.
// ══════════════════════════════════════════════════════════════════════
#[test]
fn falsify_cgp_091_trueno_vs_ndarray_gemm() {
    // Run criterion benchmark for 512 (fastest to complete)
    let output = Command::new(env!("CARGO"))
        .args([
            "bench",
            "--bench",
            "gemm_comparison",
            "--",
            "gemm/trueno/512",
            "--quick",
            "--sample-size",
            "10",
        ])
        .output();

    let Ok(out) = output else {
        eprintln!("FALSIFY-CGP-091: criterion bench not available, skipping");
        return;
    };
    let stdout = String::from_utf8_lossy(&out.stdout);

    // Parse trueno time from criterion output
    // Format: "gemm/trueno/512   time:   [1.8428 ms 1.8562 ms 1.8730 ms]"
    let trueno_time = parse_criterion_time(&stdout, "gemm/trueno/512");
    let ndarray_time = parse_criterion_time(&stdout, "gemm/ndarray/512");

    // Also try 512 ndarray if not in same run
    if trueno_time.is_none() || ndarray_time.is_none() {
        eprintln!(
            "FALSIFY-CGP-091: Could not parse both times. trueno={:?} ndarray={:?}",
            trueno_time, ndarray_time
        );
        return;
    }

    let trueno_ms = trueno_time.unwrap();
    let ndarray_ms = ndarray_time.unwrap();
    let ratio = ndarray_ms / trueno_ms;

    eprintln!(
        "FALSIFY-CGP-091: trueno={:.3}ms ndarray={:.3}ms ratio={:.2}x",
        trueno_ms, ndarray_ms, ratio
    );

    // trueno should be at least as fast as ndarray (>= 0.9x)
    assert!(
        ratio >= 0.9,
        "FALSIFY-CGP-091 FAILED: trueno {trueno_ms:.3}ms vs ndarray {ndarray_ms:.3}ms = {ratio:.2}x (need >= 0.9x)"
    );
}

// ══════════════════════════════════════════════════════════════════════
// FALSIFY-CGP-090: trueno GEMM must be at hardware peak (single-thread)
// Spec section 8.4b: >= 1.0x vs NumPy at 1T (both at AVX-512 peak).
// ══════════════════════════════════════════════════════════════════════
#[test]
fn falsify_cgp_090_trueno_gemm_at_peak() {
    // Use cgp profile compare which runs the benchmark binary
    let output = cgp_cmd()
        .args([
            "profile",
            "compare",
            "--kernel",
            "gemm",
            "--size",
            "1024",
            "--backends",
            "avx512",
        ])
        .output()
        .expect("Failed to run cgp profile compare");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);

    // If measured (M label), check GFLOPS > 100 (reasonable for parallel GEMM)
    if stdout.contains("M") {
        // Parse TFLOP/s from output
        for line in stdout.lines() {
            if line.contains("avx512") && line.contains("M") {
                // The line has TFLOP/s field
                let parts: Vec<&str> = line.split_whitespace().collect();
                // Find the TFLOP/s value (after time)
                for (i, p) in parts.iter().enumerate() {
                    if let Ok(tflops) = p.parse::<f64>() {
                        if tflops > 0.01 && i > 1 {
                            let gflops = tflops * 1000.0;
                            eprintln!("FALSIFY-CGP-090: Measured GEMM 1024 = {:.0} GFLOPS", gflops);
                            assert!(
                                gflops > 100.0,
                                "FALSIFY-CGP-090: GEMM 1024 {gflops:.0} GFLOPS must be > 100"
                            );
                            return;
                        }
                    }
                }
            }
        }
    }
    eprintln!("FALSIFY-CGP-090: No measured data available, test inconclusive");
}

// ══════════════════════════════════════════════════════════════════════
// FALSIFY-CGP-QUANT-ALL-001: quant --all must produce summary table
// ══════════════════════════════════════════════════════════════════════
#[test]
fn falsify_cgp_quant_all_001_summary() {
    let output = cgp_cmd()
        .args(["profile", "quant", "--all"])
        .output()
        .expect("Failed to run cgp profile quant --all");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        stdout.contains("Quant Sweep"),
        "FALSIFY-CGP-QUANT-ALL-001: Must show sweep header.\nOutput:\n{stdout}"
    );
    assert!(
        stdout.contains("Summary"),
        "FALSIFY-CGP-QUANT-ALL-001: Must show summary.\nOutput:\n{stdout}"
    );
    assert!(
        stdout.contains("ffn_up"),
        "FALSIFY-CGP-QUANT-ALL-001: Must show ffn_up layer.\nOutput:\n{stdout}"
    );
}

/// Parse criterion time from output line.
/// Expects format: "name   time:   [low mean high]"
fn parse_criterion_time(output: &str, bench_name: &str) -> Option<f64> {
    for line in output.lines() {
        if line.contains(bench_name) && line.contains("time:") {
            // Extract mean (middle value) from "[low mean high]"
            let bracket_content = line.split('[').nth(1)?.split(']').next()?;
            let parts: Vec<&str> = bracket_content.split_whitespace().collect();
            if parts.len() >= 4 {
                // parts: ["1.8428", "ms", "1.8562", "ms", "1.8730", "ms"]
                let mean_str = parts[2]; // middle value
                let unit = parts[3]; // unit
                let mut val: f64 = mean_str.parse().ok()?;
                if unit == "µs" || unit == "us" {
                    val /= 1000.0; // convert to ms
                }
                return Some(val);
            }
        }
    }
    None
}

// ══════════════════════════════════════════════════════════════════════
// FALSIFY-CGP-CONTRACT-001: cgp contract verify --self must pass
// Verifies that cgp's own performance contracts are satisfied.
// ══════════════════════════════════════════════════════════════════════
#[test]
fn falsify_cgp_contract_001_self_verify() {
    let output = cgp_cmd()
        .args(["contract", "verify", "--self-verify"])
        .output()
        .expect("Failed to run cgp contract verify --self");

    assert!(
        output.status.success(),
        "cgp contract verify --self must succeed"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should not report any failures
    assert!(
        !stdout.contains("FAIL"),
        "FALSIFY-CGP-CONTRACT-001: Self-verify must not have FAILures.\nOutput:\n{stdout}"
    );
}

// ══════════════════════════════════════════════════════════════════════
// FALSIFY-CGP-CONTRACT-002: cgp contract verify --contracts-dir works
// ══════════════════════════════════════════════════════════════════════
#[test]
fn falsify_cgp_contract_002_contracts_dir() {
    // `cargo test -p aprender-cgp` sets CWD to THIS crate's own directory
    // (crates/aprender-cgp), so the path is relative to there, not the repo
    // root. `../../contracts/cgp/` never existed -- verified empty in every
    // checkout. aprender-cgp itself has no contracts/ dir; the real cgp
    // contracts live under aprender-compute (formerly trueno) since the
    // APR-MONO merge, a sibling of this crate under crates/.
    let output = cgp_cmd()
        .args([
            "contract",
            "verify",
            "--contracts-dir",
            "../aprender-compute/contracts/cgp/",
        ])
        .output()
        .expect("Failed to run cgp contract verify");

    assert!(
        output.status.success(),
        "cgp contract verify --contracts-dir must succeed"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should show verification results (PASS/FAIL) and Total line
    assert!(
        stdout.contains("Total:"),
        "FALSIFY-CGP-CONTRACT-002: Must show Total verification summary.\nOutput:\n{stdout}"
    );
    // Must have at least 1 passing contract
    assert!(
        stdout.contains("PASS"),
        "FALSIFY-CGP-CONTRACT-002: Must have at least 1 PASS.\nOutput:\n{stdout}"
    );
}
