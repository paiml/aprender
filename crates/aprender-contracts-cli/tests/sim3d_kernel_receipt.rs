//! #7246 / PMAT-7246 (SIM3D P1): the receipt the determinism kernel really writes is judged by pv against
//! `contracts/sim3d-receipt-v1.yaml` (the #7245 shape), not against a hand-written copy of it.
//!
//! The receipt comes from `simular::determinism::DetSim::receipt().to_canonical_json()`, is copied into a scratch
//! corpus beside the REAL contract with an `entity:` block appended, and graded with
//! `pv lint <scratch>/contracts --gate shapes --shape sim3d-receipt-v1`. Planted defects on that same receipt
//! must go red naming the path.
//!
//! LAB: this target is not wired into `ci/explicit-test-commands.d/` (a new PR-path check is a gate change).

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;
use simular::determinism::{DetModel, DetSim, EntityState, Monitor, StepCtx};
use simular::engine::SimTime;

struct Ring;
impl DetModel for Ring {
    fn advance(&self, ctx: &StepCtx<'_>, prev: &EntityState) -> Vec<u64> {
        let r = ctx.rng.draw_u64(prev.id, ctx.step, 0);
        vec![prev.words[0].wrapping_add(r % 5), prev.words[1] ^ r]
    }
}

struct Cap(&'static str, u64);
impl Monitor for Cap {
    fn name(&self) -> &str {
        self.0
    }
    fn holds(&self, _s: u64, _t: SimTime, st: &[EntityState]) -> bool {
        st.iter().map(|e| e.words[0]).sum::<u64>() < self.1
    }
}

fn kernel_receipt(monitors: Vec<Box<dyn Monitor>>) -> String {
    let ents = (0..8)
        .map(|i| EntityState {
            id: i + 1,
            words: vec![i, 0],
        })
        .collect();
    let mut s = DetSim::new(Ring, b"spec", 42, 1_000_000, ents).expect("sim");
    for m in monitors {
        s.add_monitor(m);
    }
    s.run(50);
    s.receipt().to_canonical_json()
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
            m.values().for_each(|x| errors(x, out));
        }
        Value::Array(a) => a.iter().for_each(|x| errors(x, out)),
        _ => {}
    }
}

#[test]
fn the_kernels_canonical_receipt_passes_the_pv_shape() {
    for monitors in [
        vec![Box::new(Cap("INV-001", u64::MAX)) as Box<dyn Monitor>],
        // a violated monitor: first_bad_step arm
        vec![
            Box::new(Cap("INV-001", 20)) as Box<dyn Monitor>,
            Box::new(Cap("INV-002", u64::MAX)),
        ],
    ] {
        let doc = kernel_receipt(monitors);
        let (code, rep) = pv_judge(&doc);
        assert_eq!(code, 0, "{doc}\n{rep}");
        assert_eq!(rep["verdict"].as_str(), Some("Pass"), "{rep}");
        assert_eq!(rep["extra"]["violations"].as_u64(), Some(0), "{rep}");
        assert_eq!(rep["extra"]["focus_nodes_n"].as_u64(), Some(1), "{rep}");
    }
}

#[test]
fn planted_defects_on_the_kernels_receipt_go_red_naming_the_path() {
    let good = kernel_receipt(vec![Box::new(Cap("INV-001", u64::MAX))]);
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
    let bad_name = kernel_receipt(vec![Box::new(Cap("total-below-cap", u64::MAX))]);
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
