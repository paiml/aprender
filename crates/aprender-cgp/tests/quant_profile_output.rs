//! `profile_quant` prints to stdout and discovers the benchmark binary relative to the
//! NOTE: the label branch is `bw_pct > compute_pct => COMPUTE-BOUND`; these tests pin that behaviour as-is.
//! working directory, so the output tests re-run this test binary as a child process in a
//! scratch directory (where `cargo metadata` fails and `./target/release/examples/...` is
//! the only candidate) and assert on its stdout.

#![allow(clippy::disallowed_methods)]

use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::process::Command;

const ENV_ARGS: &str = "CGP_QUANT_CHILD_ARGS";

/// Child entry: a no-op unless the parent set the env var.
#[test]
fn child_entry() {
    let Ok(args) = std::env::var(ENV_ARGS) else {
        return;
    };
    let (kernel, size) = args.split_once(' ').expect("kernel and size");
    cgp::profilers::quant::profile_quant(kernel, size).expect("profile_quant");
}

fn run_child(cwd: &Path, kernel: &str, size: &str) -> String {
    let out = Command::new(std::env::current_exe().expect("current_exe"))
        .args(["--exact", "child_entry", "--nocapture", "--test-threads=1"])
        .env(ENV_ARGS, format!("{kernel} {size}"))
        .current_dir(cwd)
        .output()
        .expect("spawn child");
    assert!(out.status.success(), "child failed: {out:?}");
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn install_bench_shim(cwd: &Path, line: &str) {
    let dir = cwd.join("target/release/examples");
    std::fs::create_dir_all(&dir).expect("mkdir");
    let bin = dir.join("benchmark_matrix_suite");
    std::fs::write(&bin, format!("#!/bin/sh\necho '{line}'\n")).expect("write shim");
    std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).expect("chmod");
}

#[test]
fn unknown_kernel_and_bad_size_are_errors() {
    let err = cgp::profilers::quant::profile_quant("q9_gemv", "4096x1x4096").unwrap_err();
    assert!(err.to_string().contains("Unknown quant kernel"), "{err}");
    let err = cgp::profilers::quant::profile_quant("q4k", "4096x4096").unwrap_err();
    assert!(err.to_string().contains("MxNxK"), "{err}");
}

#[test]
fn without_bench_binary_prints_static_metrics_and_no_timing() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let out = run_child(tmp.path(), "q4k", "4096x1x4096");
    assert!(out.contains("Kernel: Q4kGemv"), "{out}");
    assert!(
        out.contains("Super-block: 256 elements, 144 bytes"),
        "{out}"
    );
    assert!(out.contains("Total weights: 16777216"), "{out}");
    assert!(out.contains("Super-blocks: 65536"), "{out}");
    assert!(out.contains("Compressed size: 9.44 MB"), "{out}");
    assert!(out.contains("FP32 equivalent: 67.11 MB"), "{out}");
    assert!(out.contains("Compression ratio: 7.1x"), "{out}");
    assert!(out.contains("No timing data"), "{out}");
    assert!(!out.contains("Roofline"), "{out}");
}

#[test]
fn timing_line_drives_roofline_and_token_estimate() {
    let tmp = tempfile::tempdir().expect("tempdir");
    install_bench_shim(
        tmp.path(),
        "Q4K GEMV 1536x8960 ... 100.0 us (50.0 GFLOPS, 20.0 GB/s)",
    );
    let out = run_child(tmp.path(), "q4k_gemv", "1536x1x8960");
    assert!(out.contains("Time: 100.0 us"), "{out}");
    assert!(out.contains("Compute util: 33% of AVX-512 peak"), "{out}");
    assert!(
        out.contains("Bandwidth util: 50% of practical DRAM"),
        "{out}"
    );
    assert!(out.contains("Bottleneck: COMPUTE-BOUND"), "{out}");
    assert!(out.contains("Est. 192 GEMVs/token: 19.2 ms"), "{out}");
    assert!(out.contains("Est. tokens/sec: 52.1"), "{out}");
    assert!(!out.contains("No timing data"), "{out}");
}

#[test]
fn higher_compute_than_bandwidth_pct_flips_label() {
    let tmp = tempfile::tempdir().expect("tempdir");
    install_bench_shim(
        tmp.path(),
        "Q4K GEMV 1536x8960 ... 100.0 us (90.0 GFLOPS, 10.0 GB/s)",
    );
    let out = run_child(tmp.path(), "q4k", "1536x1x8960");
    assert!(out.contains("Bottleneck: MEMORY-BOUND"), "{out}");
}

#[test]
fn timing_for_other_dimensions_is_ignored() {
    let tmp = tempfile::tempdir().expect("tempdir");
    install_bench_shim(
        tmp.path(),
        "Q4K GEMV 1536x8960 ... 100.0 us (50.0 GFLOPS, 20.0 GB/s)",
    );
    let out = run_child(tmp.path(), "q4k", "4096x1x4096");
    assert!(out.contains("No timing data"), "{out}");
}
