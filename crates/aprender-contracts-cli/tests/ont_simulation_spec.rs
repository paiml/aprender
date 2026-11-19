//! aprender#7245 (SIM3D P0, master #7244) — a simulation spec under contract (`contracts/simulation-spec-v1.yaml`).
//!
//! Each case copies the REAL contract into a scratch corpus, points an `entity:` at one fixture and has pv
//! arm the shape (the findings-ledger pattern). Widening the shape in the contract turns this table red.
//!
//! | fixture | expected |
//! |---|---|
//! | `ok` (the M/M/1 spec) | Pass |
//! | `missing-seed` | Fail, 1 violation (minCount) |
//! | `no-invariant` | Fail, 1 violation (minCount) |
//! | `extra-key` | Fail, 1 violation (closed) |
//! | `empty` | Fail, 5 violations (every required key) |

use std::path::{Path, PathBuf};
use std::process::Command;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Run pv's shapes gate on one fixture; returns (exit code, parsed report).
fn judge(name: &str) -> (i32, serde_json::Value) {
    let dir = std::env::temp_dir().join(format!("sim-spec-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("contracts")).expect("mkdir contracts");
    std::fs::create_dir_all(dir.join("data")).expect("mkdir data");
    let mut yaml = std::fs::read_to_string(root().join("contracts/simulation-spec-v1.yaml"))
        .expect("read contract");
    yaml.push_str("\nentity:\n  type: json\n  ref: data/spec.json\n");
    std::fs::write(dir.join("contracts/simulation-spec-v1.yaml"), yaml).expect("write contract");
    let fx = root().join(format!("tests/fixtures/ont/simulation-spec/{name}.json"));
    std::fs::copy(fx, dir.join("data/spec.json")).expect("copy fixture");
    let out = Command::new(env!("CARGO_BIN_EXE_pv"))
        .args(["lint"])
        .arg(dir.join("contracts"))
        .args(["--gate", "shapes", "--shape", "simulation-spec-v1"])
        .output()
        .expect("spawn pv");
    let v = serde_json::from_slice(&out.stdout).expect("pv report is JSON");
    let _ = std::fs::remove_dir_all(&dir);
    (out.status.code().unwrap_or(-1), v)
}

#[test]
fn the_mm1_spec_passes_and_the_shape_is_armed() {
    let (rc, v) = judge("ok");
    assert_eq!(rc, 0, "{v}");
    assert_eq!(v["verdict"].as_str(), Some("Pass"), "{v}");
    assert_eq!(v["extra"]["focus_nodes_n"].as_u64(), Some(1), "{v}");
    let armed = v["extra"]["armed_shapes"].to_string();
    assert!(
        armed.contains("simulation-spec-v1"),
        "vacuous pass: {armed}"
    );
}

#[test]
fn each_defective_spec_is_refused_with_its_violation_count() {
    for (name, violations) in [
        ("missing-seed", 1),
        ("no-invariant", 1),
        ("extra-key", 1),
        ("empty", 5),
    ] {
        let (rc, v) = judge(name);
        assert_eq!(rc, 1, "{name} was not refused: {v}");
        assert_eq!(v["verdict"].as_str(), Some("Fail"), "{name}: {v}");
        assert_eq!(
            v["extra"]["violations"].as_u64(),
            Some(violations),
            "{name}: {v}"
        );
    }
}
