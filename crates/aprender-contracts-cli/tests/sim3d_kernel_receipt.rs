//! #7288 (SIM3D, carried from #7246): the receipt the determinism kernel really writes is judged by pv against
//! `contracts/sim3d-receipt-v1.yaml` (the #7245 shape), not against a hand-written copy of it.
//!
//! The receipt comes from `simular::determinism::receipt::run_receipt(..)` and its `to_canonical_json()`, is
//! copied into a scratch corpus beside the REAL contract with an `entity:` block appended, and graded with
//! `pv lint <scratch>/contracts --gate shapes --shape sim3d-receipt-v1`. Planted defects on that same receipt
//! must go red naming the path.
//!
//! GATE: wired into `ci/explicit-test-commands.d/892-sim3d-kernel-receipt.cmd` (a PR-path check; this PR is a gate change, see the PR body).

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;
use simular::determinism::receipt::{run_receipt, spec_sha256, Monitor};
use simular::determinism::{mix, Entity, Spec};

/// Holds while the wrapping sum of all state words is at most the cap.
struct Cap(&'static str, u64);
impl Monitor for Cap {
    fn name(&self) -> &'static str {
        self.0
    }
    fn holds(&self, _step: u64, _t_ns: u64, state: &[Entity]) -> bool {
        state.iter().fold(0u64, |a, e| a.wrapping_add(e.state)) <= self.1
    }
}

const SPEC_BYTES: &[u8] = b"schema: sim3d-spec/v1\nname: ring\n";

fn kernel_receipt(monitors: &[&dyn Monitor]) -> String {
    let spec = Spec {
        seed: 42,
        dt_ns: 1_000_000,
        steps: 50,
        entities: (0..8)
            .map(|i| Entity {
                id: i + 1,
                state: mix(i),
            })
            .collect(),
    };
    let (_, receipt) = run_receipt(SPEC_BYTES, &spec, monitors).expect("run");
    receipt.to_canonical_json()
}

fn pv_judge(doc: &str) -> (i32, Value) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let contract = std::fs::read_to_string(root.join("contracts/sim3d-receipt-v1.yaml"))
        .expect("receipt contract readable");
    assert!(!contract.contains("\nentity:"));
    let dir = tempfile::tempdir().expect("scratch");
    std::fs::create_dir(dir.path().join("contracts")).expect("mkdir");
    std::fs::create_dir(dir.path().join("data")).expect("mkdir");
    std::fs::write(
        dir.path().join("contracts/sim3d-receipt-v1.yaml"),
        format!("{contract}\nentity:\n  type: json\n  ref: data/doc.json\n"),
    )
    .expect("write");
    std::fs::write(dir.path().join("data/doc.json"), doc).expect("write");
    let out = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_pv")))
        .arg("lint")
        .arg(dir.path().join("contracts"))
        .args(["--gate", "shapes", "--shape", "sim3d-receipt-v1"])
        .output()
        .expect("spawn pv");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let rep =
        serde_json::from_str(&stdout).unwrap_or_else(|e| panic!("report not JSON: {e}\n{stdout}"));
    (out.status.code().unwrap_or(-1), rep)
}

fn errors(v: &Value, out: &mut Vec<String>) {
    match v {
        Value::Object(m) => {
            if let (Some(msg), Some("Error")) = (
                m.get("message").and_then(Value::as_str),
                m.get("severity").and_then(Value::as_str),
            ) {
                out.push(msg.to_owned());
            }
            for x in m.values() {
                errors(x, out);
            }
        }
        Value::Array(a) => a.iter().for_each(|x| errors(x, out)),
        _ => {}
    }
}

#[test]
fn the_kernels_canonical_receipt_passes_the_pv_shape() {
    let always = Cap("INV-001", u64::MAX);
    // a violated monitor: the first_bad_step arm
    let never = Cap("INV-002", 0);
    for monitors in [
        vec![&always as &dyn Monitor],
        vec![&never as &dyn Monitor, &always],
    ] {
        let doc = kernel_receipt(&monitors);
        let (code, rep) = pv_judge(&doc);
        assert_eq!(code, 0, "{doc}\n{rep}");
        assert_eq!(rep["verdict"].as_str(), Some("Pass"), "{rep}");
        assert_eq!(rep["extra"]["violations"].as_u64(), Some(0), "{rep}");
        assert_eq!(rep["extra"]["focus_nodes_n"].as_u64(), Some(1), "{rep}");
    }
    // the violated arm really was exercised, and the receipt names the spec bytes it was given
    let v: Value = serde_json::from_str(&kernel_receipt(&[&never, &always])).expect("json");
    assert_eq!(v["verdicts"][1]["first_bad_step"].as_u64(), Some(0), "{v}");
    assert_eq!(
        v["spec_sha256"].as_str(),
        Some(spec_sha256(SPEC_BYTES).as_str())
    );
}

#[test]
fn planted_defects_on_the_kernels_receipt_go_red_naming_the_path() {
    let always = Cap("INV-001", u64::MAX);
    let good = kernel_receipt(&[&always]);
    let mut v: Value = serde_json::from_str(&good).expect("json");
    let o = v.as_object_mut().expect("object");
    o.remove("final_hash");
    let no_hash = serde_json::to_string(&v).expect("json");
    let mut v: Value = serde_json::from_str(&good).expect("json");
    v.as_object_mut()
        .expect("o")
        .insert("verdicts".into(), Value::Array(vec![]));
    let no_verdicts = serde_json::to_string(&v).expect("json");
    let mut v: Value = serde_json::from_str(&good).expect("json");
    v.as_object_mut()
        .expect("o")
        .insert("host".into(), Value::from("x"));
    let extra_key = serde_json::to_string(&v).expect("json");
    // a monitor whose name is not an INV-nnn id is refused by the shape, not by the kernel
    let bad = Cap("total-below-cap", u64::MAX);
    let bad_name = kernel_receipt(&[&bad]);
    for (doc, path) in [
        (no_hash, "ont:sim/final_hash"),
        (no_verdicts, "ont:sim/verdicts"),
        (extra_key, "host"),
        (bad_name, "ont:sim/invariant"),
    ] {
        let (code, rep) = pv_judge(&doc);
        assert_eq!(code, 1, "accepted: {doc}\n{rep}");
        let mut errs = Vec::new();
        errors(&rep, &mut errs);
        assert!(
            errs.iter().any(|m| m.contains(path)),
            "want `{path}` in {errs:#?}"
        );
    }
}

/// `sha256sum <spec>` equals the receipt's `spec_sha256`, for a real spec file on main.
#[test]
fn sha256sum_of_the_spec_file_equals_the_receipts_spec_sha256() {
    let spec_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/ont/sim3d/mm1-littles-law.sim.yaml");
    let bytes = std::fs::read(&spec_path).expect("spec fixture readable");
    assert_ne!(bytes.len(), 0);
    let out = Command::new("sha256sum")
        .arg(&spec_path)
        .output()
        .expect("spawn sha256sum");
    assert!(out.status.success(), "sha256sum failed: {out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let want = stdout.split_whitespace().next().expect("a digest");
    assert_eq!(want.len(), 64, "{stdout}");
    let spec = Spec {
        seed: 42,
        dt_ns: 1_000_000,
        steps: 4,
        entities: vec![Entity { id: 1, state: 0 }],
    };
    let always = Cap("INV-001", u64::MAX);
    let (_, receipt) = run_receipt(&bytes, &spec, &[&always]).expect("run");
    assert_eq!(receipt.spec_sha256, want);
    let v: Value = serde_json::from_str(&receipt.to_canonical_json()).expect("json");
    assert_eq!(v["spec_sha256"].as_str(), Some(want));
}
