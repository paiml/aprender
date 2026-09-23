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

// ── The row's own probe on THIS repository, and the row's data mutations (each must turn it RED) ──────────────
//
// The probe is ONT-4c5's jq predicate, restated: verdict Pass, `capability-cells` armed, `pc_shapes` fired, no
// NotRun cell, and every id of the baseline set B in the domain. The mutations run in a scratch tree that is
// symlinks all the way down except on the path to the one file a mutation edits, which is a real copy — the
// checkout is never written. The row's code mutation ("fold a missing cell into Fail") is held by the unit tests
// and the plant, not here.

/// ONT-4c5's baseline set B, verbatim from the row's probe (paiml/infra paiml-ontology.md v4.12 §5).
const B: [&str; 14] = [
    "qwen2-1.5b-q4km@lambda",
    "qwen2-1.5b-q4km@gx10",
    "qwen3-1.7b-q4km@lambda",
    "qwen3-1.7b-q4km@gx10",
    "qwen35-0.8b-q4km@lambda",
    "qwen35-0.8b-q4km@gx10",
    "qwen35-2b-q4km@lambda",
    "qwen35-2b-q4km@gx10",
    "qwen35-4b-q4km@lambda",
    "qwen35-4b-q4km@gx10",
    "qwen35-9b-q4km@lambda",
    "qwen35-9b-q4km@gx10",
    "qwen35-27b-q4km@lambda",
    "qwen35-27b-q4km@gx10",
];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn lint_root(root: &Path) -> Run {
    let out = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_pv")))
        .current_dir(root)
        .args(["lint", "contracts", "--gate", "shapes", "--format", "json"])
        .output()
        .expect("failed to spawn pv");
    Run {
        code: out.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

/// The row's probe predicate over one run: `Ok(())` green, `Err(why)` red.
fn probe(r: &Run) -> Result<(), String> {
    let v: serde_json::Value =
        serde_json::from_str(&r.stdout).map_err(|_| format!("no JSON (exit {})", r.code))?;
    let has = |key: &str, id: &str| {
        v[key]
            .as_array()
            .is_some_and(|a| a.iter().any(|x| x.as_str() == Some(id)))
    };
    if v["verdict"] != "Pass" {
        return Err(format!("verdict {}", v["verdict"]));
    }
    if !has("armed_shapes", "capability-cells") {
        return Err("capability-cells not armed".into());
    }
    if v["pc_shapes"]["capability-cells"] != "fired" {
        return Err("pc_shapes not fired".into());
    }
    let cc = &v["capability_cells"];
    if cc["not_run"].as_array().is_none_or(|a| !a.is_empty()) {
        return Err(format!("not_run {}", cc["not_run"]));
    }
    let domain: Vec<&str> = cc["domain"]
        .as_array()
        .map(|a| a.iter().filter_map(serde_json::Value::as_str).collect())
        .unwrap_or_default();
    let missing: Vec<&str> = B.iter().copied().filter(|b| !domain.contains(b)).collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!("B ⊄ D: {missing:?}"))
    }
}

/// A scratch copy of the repository in which `rel` (and only it) is a real, writable file.
fn scratch_with(rel: &str) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("scratch");
    let repo = repo_root().canonicalize().expect("repo root");
    let rel = Path::new(rel);
    let mut src = repo.clone();
    let mut dst = dir.path().to_path_buf();
    let comps: Vec<_> = rel.components().collect();
    for (i, comp) in comps.iter().enumerate() {
        // at this level: symlink every sibling, descend into (or copy) the one on the path
        for entry in std::fs::read_dir(&src).expect("read_dir") {
            let entry = entry.expect("entry");
            let name = entry.file_name();
            if name == comp.as_os_str() || name == ".git" || name == "target" || name == ".pv" {
                continue;
            }
            std::os::unix::fs::symlink(entry.path(), dst.join(&name)).expect("symlink");
        }
        src = src.join(comp);
        dst = dst.join(comp);
        if i + 1 == comps.len() {
            std::fs::copy(&src, &dst).expect("copy the mutated file");
        } else {
            std::fs::create_dir(&dst).expect("mkdir on the path");
        }
    }
    (dir, dst)
}

fn edit_json(path: &Path, f: impl FnOnce(&mut serde_json::Value)) {
    let mut v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).expect("read")).expect("json");
    f(&mut v);
    std::fs::write(path, serde_json::to_string_pretty(&v).expect("ser")).expect("write");
}

fn rung_row<'a>(v: &'a mut serde_json::Value, id: &str) -> &'a mut serde_json::Value {
    v["rungs"]
        .as_array_mut()
        .expect("rungs")
        .iter_mut()
        .find(|r| r["id"] == id)
        .unwrap_or_else(|| panic!("mutation anchor missing: {id}"))
}

#[test]
fn the_rows_probe_is_green_on_this_repository() {
    let r = lint_root(&repo_root());
    probe(&r).unwrap_or_else(|why| panic!("ONT-4c5 probe RED on the tree: {why}\n{}", show(&r)));
}

#[test]
fn disarming_capability_cells_turns_the_probe_red() {
    let (dir, f) = scratch_with("contracts/lint-baseline.json");
    edit_json(&f, |v| {
        let a = v["armed_shapes"].as_array_mut().expect("armed_shapes");
        let n = a.len();
        a.retain(|x| x != "capability-cells");
        assert_eq!(a.len(), n - 1, "mutation anchor missing");
    });
    let r = lint_root(dir.path());
    assert!(probe(&r).is_err(), "disarmed and still green\n{}", show(&r));
    assert_eq!(
        r.code,
        2,
        "the plant cannot fire through an unarmed shape\n{}",
        show(&r)
    );
}

#[test]
fn flipping_a_required_rung_to_optional_leaves_b_outside_the_domain() {
    let (dir, f) = scratch_with("contracts/model-capability-ladder-v1.yaml");
    let text = std::fs::read_to_string(&f).expect("read");
    let at = text
        .find("- id: qwen35-2b-q4km")
        .expect("mutation anchor missing");
    let tail = &text[at..];
    let next = tail[1..].find("\n    - id: ").map_or(tail.len(), |n| n + 1);
    let off = tail[..next]
        .find("required: true")
        .expect("the rung has its own required: true");
    let mut out = text.clone();
    out.replace_range(
        at + off..at + off + "required: true".len(),
        "required: false",
    );
    std::fs::write(&f, out).expect("write");
    let r = lint_root(dir.path());
    let why = probe(&r).expect_err("an optional rung left B ⊆ D intact");
    assert!(why.contains("qwen35-2b-q4km@lambda"), "{why}");
}

#[test]
fn a_not_run_label_on_a_current_release_row_rejects_naming_the_cell() {
    for label in ["DEFER", "MANUAL", "NO-VERDICT"] {
        let (dir, f) = scratch_with("evidence/dogfood/models/0.69.1/lambda.json");
        edit_json(&f, |v| {
            rung_row(v, "qwen35-4b-q4km")["verdict"] = label.into()
        });
        let r = lint_root(dir.path());
        assert_eq!(r.code, 1, "{label}\n{}", show(&r));
        assert!(probe(&r).is_err(), "{label}");
        assert!(
            r.stdout.contains("qwen35-4b-q4km@lambda"),
            "{label}\n{}",
            show(&r)
        );
    }
}

#[test]
fn a_deleted_current_release_row_rejects_naming_the_cell() {
    let (dir, f) = scratch_with("evidence/dogfood/models/0.69.1/gx10.json");
    edit_json(&f, |v| {
        let rows = v["rungs"].as_array_mut().expect("rungs");
        let n = rows.len();
        rows.retain(|r| r["id"] != "qwen3-1.7b-q4km");
        assert_eq!(rows.len(), n - 1, "mutation anchor missing");
    });
    let r = lint_root(dir.path());
    assert_eq!(r.code, 1, "{}", show(&r));
    assert!(r.stdout.contains("qwen3-1.7b-q4km@gx10"), "{}", show(&r));
}
