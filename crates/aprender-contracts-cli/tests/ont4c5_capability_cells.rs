//! ONT-4c5 (PMAT-3972; aprender #3972 + #4047) — capability cells, on the CLI.
//!
//! ONT-001 v4.12 §5 ONT-4c5 RED, on `pv lint --gate shapes`: a required cell of the model-capability ladder is
//! Pass or Fail at the current release, and `Unknown{NotRun}` — a missing row, DEFER, MANUAL, NO-VERDICT, a label
//! nobody knows — is a violation of the ARMED `capability-cells` shape: exit 1, naming the cell. An empty domain
//! is a decline (exit 2), never Pass. A Fail cell is ADMITTED by this shape (ladder-green is what refuses it).
//!
//! DISCRIMINATION: `capcells-clean/` must PASS with `pc_shapes` fired and the whole domain reported, so a build
//! that declines or rejects every ladder fails this file; each NotRun fixture must FAIL naming exactly its cell,
//! so a build that arms nothing, or folds an absence into Fail, fails it too; `capcells-failadmit/` must PASS,
//! so a build that treats Fail as NotRun fails it.

use std::path::{Path, PathBuf};
use std::process::Command;

struct Run {
    code: i32,
    stdout: String,
    stderr: String,
}

fn lint(fixture: &str) -> Run {
    let contracts = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/ont")
        .join(fixture)
        .join("contracts");
    let scratch = tempfile::tempdir().expect("scratch cwd is creatable");
    let out = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_pv")))
        .current_dir(scratch.path())
        .args([
            "lint",
            &contracts.to_string_lossy(),
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

fn show(r: &Run) -> String {
    format!(
        "exit {}\n--- stdout\n{}\n--- stderr\n{}",
        r.code, r.stdout, r.stderr
    )
}

fn json(r: &Run) -> serde_json::Value {
    serde_json::from_str(&r.stdout).unwrap_or_else(|e| panic!("not JSON ({e}):\n{}", show(r)))
}

fn strings(v: &serde_json::Value) -> Vec<String> {
    v.as_array()
        .unwrap_or_else(|| panic!("not an array: {v}"))
        .iter()
        .map(|x| x.as_str().expect("string").to_string())
        .collect()
}

const DOMAIN: [&str; 4] = [
    "rung-a@gx10",
    "rung-a@lambda",
    "rung-b@gx10",
    "rung-b@lambda",
];

#[test]
fn a_clean_ladder_passes_with_the_plant_fired_and_the_whole_domain_reported() {
    let r = lint("capcells-clean");
    assert_eq!(r.code, 0, "{}", show(&r));
    let v = json(&r);
    assert_eq!(v["verdict"], "Pass", "{}", show(&r));
    assert_eq!(v["pc_shapes"]["capability-cells"], "fired", "{}", show(&r));
    assert!(strings(&v["armed_shapes"]).contains(&"capability-cells".to_string()));
    assert_eq!(strings(&v["capability_cells"]["domain"]), DOMAIN);
    assert!(strings(&v["capability_cells"]["not_run"]).is_empty());
    assert_eq!(v["capability_cells"]["v_star"], "0.69.1");
}

/// Every NotRun fixture: exit 1, `not_run` is exactly its cell, and a finding names that cell.
#[test]
fn every_not_run_required_cell_rejects_naming_exactly_that_cell() {
    for (fixture, cell) in [
        ("capcells-defer", "rung-a@lambda"),
        ("capcells-manual", "rung-a@gx10"),
        ("capcells-noverdict", "rung-b@lambda"),
        ("capcells-missing", "rung-b@gx10"),
        ("capcells-badlabel", "rung-a@lambda"),
    ] {
        let r = lint(fixture);
        assert_eq!(r.code, 1, "{fixture}: {}", show(&r));
        let v = json(&r);
        assert_eq!(v["verdict"], "Fail", "{fixture}");
        assert_eq!(v["pc_shapes"]["capability-cells"], "fired", "{fixture}");
        assert_eq!(
            strings(&v["capability_cells"]["not_run"]),
            [cell],
            "{fixture}"
        );
        let named = v["findings"].as_array().expect("findings").iter().any(|f| {
            f["rule_id"] == "PV-ONT-013"
                && f["message"]
                    .as_str()
                    .is_some_and(|m| m.contains(&format!("capability cell {cell} ")))
        });
        assert!(
            named,
            "{fixture}: no PV-ONT-013 finding names {cell}\n{}",
            show(&r)
        );
    }
}

#[test]
fn an_unknown_label_is_refused_by_name() {
    let r = lint("capcells-badlabel");
    let v = json(&r);
    let refused = strings(&v["capability_cells"]["refused_labels"]);
    assert_eq!(refused.len(), 1, "{}", show(&r));
    assert!(refused[0].ends_with(": rung-a: DEFERRED"), "{refused:?}");
}

#[test]
fn a_fail_cell_is_admitted_by_this_shape() {
    let r = lint("capcells-failadmit");
    assert_eq!(r.code, 0, "{}", show(&r));
    let v = json(&r);
    assert_eq!(v["verdict"], "Pass");
    assert!(strings(&v["capability_cells"]["not_run"]).is_empty());
}

#[test]
fn an_empty_domain_declines_never_passes() {
    let r = lint("capcells-emptydomain");
    assert_eq!(r.code, 2, "{}", show(&r));
    assert!(
        r.stderr.contains("capability-cells domain D is empty"),
        "{}",
        show(&r)
    );
    assert!(r.stderr.contains("decline:"), "{}", show(&r));
}
