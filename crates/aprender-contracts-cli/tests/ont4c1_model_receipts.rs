//! ONT-4c1 (PMAT-3508) — model receipts as focus nodes, on the CLI.
//!
//! ONT-001 v4.6 §5 ONT-4c1 RED, on `pv lint --gate shapes` and `pv extract`: the ladder rungs are `model:Model`
//! nodes; `resolves: receipt` joins the tracked receipts by measured sha256 (a row without `sha256` is not a
//! witness, a different hex is a reject naming file and hex); `ladder-measured` is armed, `ladder-green` is
//! computed and reported; arming per shape is monotone (`armed_shapes shrank` → exit 3); no receipt file at all
//! is a decline naming the directory; a foreign receipt schema is exit 3 naming the file; a lying `.apr` header
//! is a reject naming the file; the report carries `armed_shapes`, `not_armed_shapes`, `by_entity_type`,
//! `pc_extract`.
//!
//! DISCRIMINATION: `ladder-green/` must PASS with 4 witnesses and the plant fired, so a build that declines every
//! ladder corpus fails this file; `ladder-fallback-armed/` must FAIL naming `gx10`, so a build that arms nothing
//! fails it too.

use std::path::{Path, PathBuf};
use std::process::Command;

fn pv_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pv"))
}

struct Run {
    code: i32,
    stdout: String,
    stderr: String,
}

fn pv_in(cwd: &Path, args: &[&str]) -> Run {
    let out = Command::new(pv_bin())
        .current_dir(cwd)
        .args(args)
        .output()
        .expect("failed to spawn pv");
    Run {
        code: out.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

fn pv(args: &[&str]) -> Run {
    let scratch = tempfile::tempdir().expect("scratch cwd is creatable");
    pv_in(scratch.path(), args)
}

fn show(r: &Run) -> String {
    format!(
        "exit {}\n--- stdout\n{}\n--- stderr\n{}",
        r.code, r.stdout, r.stderr
    )
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/ont")
        .join(name)
        .join("contracts")
}

fn s(p: &Path) -> String {
    p.to_str().expect("utf-8 path").to_string()
}

fn json_of(r: &Run) -> serde_json::Value {
    serde_json::from_str(&r.stdout).unwrap_or_else(|e| panic!("stdout is JSON: {e}\n{}", show(r)))
}

fn gate(dir: &Path) -> Run {
    pv(&["lint", &s(dir), "--gate", "shapes", "--format", "json"])
}

#[test]
fn an_all_green_ladder_passes_with_measured_armed_green_reported_and_both_extract_controls_fired() {
    let r = gate(&fixture("ladder-green"));
    assert_eq!(r.code, 0, "{}", show(&r));
    let v = json_of(&r);
    assert_eq!(v["verdict"], "Pass", "{}", show(&r));
    assert_eq!(v["pc_shape"], "fired");
    assert_eq!(v["pc_extract"]["gguf"], "fired", "{}", show(&r));
    assert_eq!(v["pc_extract"]["apr-model"], "fired", "{}", show(&r));
    assert!(
        v["by_entity_type"]["gguf"].as_u64().unwrap_or(0) >= 1,
        "{}",
        show(&r)
    );
    let armed: Vec<&str> = v["armed_shapes"]
        .as_array()
        .map(|a| a.iter().filter_map(|x| x.as_str()).collect())
        .unwrap_or_default();
    assert!(armed.contains(&"ladder-measured"), "{}", show(&r));
    assert_eq!(
        v["not_armed_shapes"].as_array().map(Vec::len),
        Some(1),
        "{}",
        show(&r)
    );
    assert_eq!(v["not_armed_shapes"][0], "ladder-green", "{}", show(&r));
    assert_eq!(v["witnesses"], 4, "two rungs × two hosts\n{}", show(&r));
    assert!(
        v["by_shape"].as_array().is_some_and(|a| a
            .iter()
            .any(|x| x.as_str().is_some_and(|x| x.starts_with("ladder-green=")))),
        "ladder-green is reported by_shape\n{}",
        show(&r)
    );
}

#[test]
fn a_fallback_host_is_reported_by_the_unarmed_shape_and_the_exit_is_still_0() {
    let r = gate(&fixture("ladder-fallback"));
    assert_eq!(r.code, 0, "{}", show(&r));
    let v = json_of(&r);
    assert_eq!(v["verdict"], "Pass");
    assert_eq!(v["unarmed_violations"], 1, "{}", show(&r));
    assert!(r.stdout.contains("[not armed]"), "{}", show(&r));
    assert!(r.stdout.contains("gx10"), "names the host\n{}", show(&r));
    assert!(
        r.stdout.contains("rung rung-b"),
        "names the rung\n{}",
        show(&r)
    );
}

#[test]
fn arming_the_green_shape_turns_the_same_fallback_into_a_reject_naming_rung_and_host() {
    let r = gate(&fixture("ladder-fallback-armed"));
    assert_eq!(r.code, 1, "{}", show(&r));
    assert_eq!(json_of(&r)["verdict"], "Fail");
    assert!(r.stdout.contains("ladder-green"), "{}", show(&r));
    assert!(r.stdout.contains("gx10"), "{}", show(&r));
    assert!(r.stdout.contains("rung rung-b"), "{}", show(&r));
}

#[test]
fn a_different_hex_in_a_receipt_row_rejects_ladder_measured_naming_the_file() {
    let r = gate(&fixture("ladder-wronghex"));
    assert_eq!(r.code, 1, "{}", show(&r));
    assert!(r.stdout.contains("receiptHexMismatch"), "{}", show(&r));
    assert!(r.stdout.contains("lambda.json"), "{}", show(&r));
    assert!(r.stdout.contains("cccccccc"), "names the hex\n{}", show(&r));
    assert_eq!(json_of(&r)["hex_mismatches"], 1);
}

#[test]
fn rows_without_sha256_are_not_witnesses_so_the_old_receipts_reject_measured() {
    let r = gate(&fixture("ladder-nosha"));
    assert_eq!(r.code, 1, "{}", show(&r));
    let v = json_of(&r);
    assert_eq!(v["witnesses"], 0);
    assert_eq!(v["unmeasured_rows"], 2);
    assert!(r.stdout.contains("parityReceipt"), "{}", show(&r));
}

#[test]
fn no_receipt_file_at_all_declines_at_exit_2_naming_the_directory() {
    let r = gate(&fixture("ladder-noreceipts"));
    assert_eq!(r.code, 2, "{}", show(&r));
    assert!(r.stderr.contains("decline: NoCheckable"), "{}", show(&r));
    assert!(
        r.stderr.contains("evidence/dogfood/models"),
        "the WHY travels with the decline\n{}",
        show(&r)
    );
}

#[test]
fn a_foreign_receipt_schema_is_an_error_at_exit_3_naming_the_file() {
    let r = gate(&fixture("ladder-badschema"));
    assert_eq!(r.code, 3, "{}", show(&r));
    assert!(r.stderr.contains("lambda.json"), "{}", show(&r));
    assert!(r.stderr.contains("refused by name"), "{}", show(&r));
}

#[test]
fn a_lying_apr_header_is_a_reject_naming_the_file_and_the_counts() {
    let r = gate(&fixture("ladder-lyingapr"));
    assert_eq!(r.code, 1, "{}", show(&r));
    assert!(r.stdout.contains("PV-ONT-012"), "{}", show(&r));
    assert!(r.stdout.contains("lying.apr"), "{}", show(&r));
    assert!(r.stdout.contains("header says 3"), "{}", show(&r));
}

#[test]
fn a_plant_only_an_unarmed_shape_can_catch_is_a_positive_control_failure() {
    let r = gate(&fixture("ladder-plantunarmed"));
    assert_eq!(r.code, 2, "{}", show(&r));
    assert!(
        r.stderr.contains("decline: PositiveControlFailed"),
        "{}",
        show(&r)
    );
}

/// `armed_shapes shrank` → exit 3, through the same comparand machinery as `armed_gates`: a scratch git repo
/// whose committed baseline arms two shapes and whose working tree arms one.
#[test]
fn dropping_an_armed_shape_against_the_committed_baseline_is_exit_3() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let repo = tmp.path();
    let git = |args: &[&str]| {
        let out = Command::new("git")
            .arg("-C")
            .arg(repo)
            .args(args)
            .env_remove("GIT_DIR")
            .env_remove("GIT_WORK_TREE")
            .env_remove("GIT_INDEX_FILE")
            .output()
            .expect("git runs");
        assert!(
            out.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    };
    git(&["init", "-q", "-b", "main"]);
    git(&["config", "user.email", "t@example.com"]);
    git(&["config", "user.name", "t"]);
    // the fixture corpus, committed with both ladder shapes armed
    let src = fixture("ladder-green");
    let dst = repo.join("contracts");
    std::fs::create_dir_all(&dst).expect("contracts dir");
    for e in std::fs::read_dir(&src).expect("fixture dir") {
        let p = e.expect("entry").path();
        std::fs::copy(&p, dst.join(p.file_name().expect("name"))).expect("copy");
    }
    std::fs::write(
        dst.join("lint-baseline.json"),
        "{\n  \"armed_gates\": [\"shapes\"],\n  \"armed_shapes\": [\"ladder-measured\", \"ladder-green\"]\n}\n",
    )
    .expect("baseline");
    git(&["add", "-A"]);
    git(&["commit", "-q", "-m", "committed: both armed"]);
    // the working tree drops one
    std::fs::write(
        dst.join("lint-baseline.json"),
        "{\n  \"armed_gates\": [\"shapes\"],\n  \"armed_shapes\": [\"ladder-measured\"]\n}\n",
    )
    .expect("baseline");
    let r = pv_in(repo, &["lint", "contracts", "--armed-baseline-ref", "HEAD"]);
    assert_eq!(r.code, 3, "{}", show(&r));
    assert!(
        r.stderr.contains("armed_shapes shrank: ladder-green"),
        "{}",
        show(&r)
    );
    // and adding one back is not a shrink
    std::fs::write(
        dst.join("lint-baseline.json"),
        "{\n  \"armed_gates\": [\"shapes\"],\n  \"armed_shapes\": [\"ladder-measured\", \"ladder-green\", \"ladder-extra\"]\n}\n",
    )
    .expect("baseline");
    let r = pv_in(repo, &["lint", "contracts", "--armed-baseline-ref", "HEAD"]);
    assert_ne!(r.code, 3, "{}", show(&r));
}

/// R-15/R-18 on the widened graph: `pv extract` writes the model nodes, twice identically, and `--check` sees drift.
#[test]
fn extract_writes_model_nodes_deterministically() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path();
    let dir = root.join("contracts");
    std::fs::create_dir_all(&dir).expect("contracts dir");
    let src = fixture("ladder-green");
    for e in std::fs::read_dir(&src).expect("fixture dir") {
        let p = e.expect("entry").path();
        std::fs::copy(&p, dir.join(p.file_name().expect("name"))).expect("copy");
    }
    let ev = root.join("evidence/dogfood/models/0.68.1");
    std::fs::create_dir_all(&ev).expect("evidence dir");
    for host in ["lambda", "gx10"] {
        std::fs::copy(
            src.parent()
                .expect("fixture root")
                .join(format!("evidence/dogfood/models/0.68.1/{host}.json")),
            ev.join(format!("{host}.json")),
        )
        .expect("copy receipt");
    }
    let d = s(&dir);
    let a = pv(&["extract", &d]);
    assert_eq!(a.code, 0, "{}", show(&a));
    let nt1 = std::fs::read_to_string(dir.join("contracts.nt")).expect("contracts.nt written");
    assert!(
        nt1.contains("v1alpha1/model/"),
        "model nodes are in the graph"
    );
    assert!(
        nt1.contains("model/parityReceipt"),
        "receipt edges are in the graph"
    );
    assert!(!nt1.contains("_:"));
    let b = pv(&["extract", &d]);
    assert_eq!(b.code, 0);
    assert_eq!(
        nt1,
        std::fs::read_to_string(dir.join("contracts.nt")).expect("rewritten")
    );
    assert_eq!(json_of(&a)["sha256"], json_of(&b)["sha256"]);
    assert_eq!(json_of(&a)["shapes_n"], 2);
}

/// The real corpus: the rungs are focus nodes, the two extract controls fire, ladder-measured is armed and
/// ladder-green is reported — whatever the receipts on this tree say about green.
#[test]
fn the_repo_corpus_reports_the_ladder_rungs_as_focus_nodes_with_the_controls_fired() {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts");
    let r = gate(&repo);
    let v = json_of(&r);
    assert!(
        v["by_entity_type"]["gguf"].as_u64().unwrap_or(0) >= 4,
        "{}",
        show(&r)
    );
    assert_eq!(v["pc_extract"]["gguf"], "fired", "{}", show(&r));
    assert_eq!(v["pc_extract"]["apr-model"], "fired", "{}", show(&r));
    let armed: Vec<&str> = v["armed_shapes"]
        .as_array()
        .map(|a| a.iter().filter_map(|x| x.as_str()).collect())
        .unwrap_or_default();
    assert!(
        armed.contains(&"ladder-measured") && armed.contains(&"ont-shapes-v1"),
        "{}",
        show(&r)
    );
    assert_eq!(
        v["not_armed_shapes"].as_array().map(Vec::len),
        Some(1),
        "{}",
        show(&r)
    );
    assert_eq!(v["not_armed_shapes"][0], "ladder-green", "{}", show(&r));
    assert!(v["receipts"].as_u64().unwrap_or(0) >= 1, "{}", show(&r));
}
