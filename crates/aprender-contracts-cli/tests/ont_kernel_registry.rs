//! KREG-001 / aprender#4539 — the kernel registry's shape, on the CLI.
//!
//! **Both arms.** A shape that only ever sees conforming rows is indistinguishable from `exit 0`, so
//! the table plants the two defects the registry exists to make unloadable and requires RED on each.
//!
//! | fixture | mutation of row 0 | expected |
//! |---|---|---|
//! | `kreg-ok` | none (3 rows: cpu/cuda/wgpu) | **Pass**, 3 focus nodes |
//! | `kreg-colmajor` | `layout: col_major` | Fail — LAYOUT-001, FALSIFY-KREG-001 |
//! | `kreg-no-qtype` | `qtype` absent | Fail — the raw_q4k class, FALSIFY-KREG-002 |
//! | `kreg-unknown-qtype` | `qtype: Q4K_RAW` | Fail — a qtype outside the closed set |
//! | `kreg-typed-tolerance` | `tolerance: 1e-3` | Fail — a number with no receipt, FALSIFY-KREG-003 |

use std::path::{Path, PathBuf};
use std::process::Command;

fn pv_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pv"))
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/ont")
        .join(name)
}

struct Run {
    code: i32,
    stdout: String,
    stderr: String,
}

impl Run {
    fn all(&self) -> String {
        format!(
            "exit {}\n--- stdout\n{}\n--- stderr\n{}",
            self.code, self.stdout, self.stderr
        )
    }
}

fn shapes_on(name: &str) -> Run {
    let contracts = fixture(name).join("contracts");
    let out = Command::new(pv_bin())
        .args([
            "lint",
            contracts.to_str().expect("utf-8 path"),
            "--gate",
            "shapes",
            "--format",
            "json",
        ])
        .output()
        .expect("failed to spawn pv");
    Run {
        code: out.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

/// Refused, by the kernel-registry shape, naming row 0 and the property that fired.
fn assert_refused(name: &str, property: &str) {
    let r = shapes_on(name);
    assert_eq!(r.code, 1, "{name} was not refused\n{}", r.all());
    assert!(
        r.stdout.contains("kernel-registry-v1.kernels.0"),
        "{name}: the report must name the focus node\n{}",
        r.all()
    );
    assert!(
        r.stdout.contains(&format!("ont:kreg/{property}")),
        "{name}: the report must name the property `{property}`\n{}",
        r.all()
    );
}

#[test]
fn a_conforming_registry_passes_with_every_row_a_focus_node() {
    let r = shapes_on("kreg-ok");
    assert_eq!(r.code, 0, "{}", r.all());
    let v: serde_json::Value = serde_json::Deserializer::from_str(&r.stdout)
        .into_iter()
        .next()
        .expect("a json report")
        .expect("json report");
    assert_eq!(v["extra"]["violations"].as_u64(), Some(0), "{}", r.all());
    assert!(
        v["extra"]["by_shape"]
            .as_array()
            .expect("by_shape")
            .iter()
            .any(|s| s.as_str() == Some("kernel-registry-v1=3")),
        "all three rows must be focus nodes\n{}",
        r.all()
    );
}

#[test]
fn falsify_kreg_001_a_col_major_row_is_refused() {
    assert_refused("kreg-colmajor", "layout");
}

#[test]
fn falsify_kreg_002_a_row_with_no_qtype_is_refused() {
    assert_refused("kreg-no-qtype", "qtype");
}

#[test]
fn a_qtype_outside_the_closed_set_is_refused() {
    assert_refused("kreg-unknown-qtype", "qtype");
}

#[test]
fn falsify_kreg_003_a_tolerance_with_no_receipt_is_refused() {
    assert_refused("kreg-typed-tolerance", "tolerance");
}
