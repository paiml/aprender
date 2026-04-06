//! Ground-truth end-to-end tests for `pv equations` across all 4 output
//! formats (text, latex, ptx, asm) and 5 deterministic fixture contracts
//! (relu, clamp, dot, scale, l2norm).
//!
//! Each test verifies exact byte-for-byte output against golden files in
//! `tests/fixtures/expected/`. The provability cross-checks verify that
//! every equation, invariant, phase, and proof obligation from the YAML
//! contract is faithfully represented across all output formats.

use std::path::{Path, PathBuf};
use std::process::Command;

fn pv_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pv"))
}

fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn run_equations(contract: &str, format: &str) -> String {
    let output = Command::new(pv_bin())
        .arg("equations")
        .arg(fixture_path(contract))
        .arg("--format")
        .arg(format)
        .output()
        .expect("failed to run pv");
    assert!(
        output.status.success(),
        "pv equations --format {format} failed for {contract}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

// Part 1: Golden-file comparison tests
include!("includes/ground_truth_golden.rs");

// Part 2: Cross-format provability checks
include!("includes/ground_truth_provability.rs");
