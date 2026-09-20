//! ONT-4b2 (PMAT-3509) — bound Rust symbols and in-tree Lean theorems as focus nodes, on the CLI.
//!
//! ONT-001 §5 ONT-4b2 RED, through `pv lint --gate shapes` and `pv extract`: the bound symbols of every
//! `binding.yaml` are `ont:Symbol` nodes under the `symbol/` IRI root, resolved by the `syn` walk or carrying
//! the reason they were not; the in-tree theorems are `ont:Statement` nodes under `world/`, and a file with
//! `sorry` grounds nothing; both arms report their counts; the vendored W3C SHACL-Core cases run every time
//! and a failing one declines `Unknown{Differential}` instead of grading the corpus.
//!
//! DISCRIMINATION, both directions: `code-ghost/` must FAIL naming the unresolved symbol (a build that drops
//! what it cannot resolve passes it, silently), and `code-bound/` — the SAME ghost with the shape unarmed —
//! must PASS with the violation reported under `unarmed_violations` (a build that ignores arming fails it).

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
}

fn json_of(r: &Run) -> serde_json::Value {
    serde_json::from_str(&r.stdout).unwrap_or_else(|e| panic!("stdout is JSON: {e}\n{}", show(r)))
}

/// The gate, run FROM the fixture root: `extract:code` resolves the workspace at the contract dir's parent and
/// `extract:lean` looks for the Lean tree there, so the cwd is part of the fixture.
fn gate(name: &str) -> Run {
    pv_in(
        &fixture(name),
        &["lint", "contracts", "--gate", "shapes", "--format", "json"],
    )
}

#[test]
fn the_bound_symbols_are_focus_nodes_and_the_walk_reports_what_it_resolved() {
    let r = gate("code-bound");
    assert_eq!(r.code, 0, "{}", show(&r));
    let v = json_of(&r);
    assert_eq!(v["verdict"], "Pass", "{}", show(&r));
    assert_eq!(v["by_entity_type"]["code"], 6, "{}", show(&r));
    assert_eq!(v["symbols_resolved"], 5, "{}", show(&r));
    assert_eq!(v["symbols_unresolved"], 1, "{}", show(&r));
    assert_eq!(v["pc_extract"]["code"], "fired", "{}", show(&r));
    assert_eq!(v["pc_extract"]["lean"], "fired", "{}", show(&r));
}

#[test]
fn an_unarmed_shape_reports_the_ghost_and_an_armed_one_fails_on_it() {
    // The same ghost binding, the same shape, two arming declarations.
    let reported = gate("code-bound");
    let vr = json_of(&reported);
    assert_eq!(reported.code, 0, "{}", show(&reported));
    assert_eq!(vr["unarmed_violations"], 1, "{}", show(&reported));
    assert_eq!(vr["violations"], 0, "{}", show(&reported));
    assert!(
        vr["not_armed_shapes"]
            .as_array()
            .is_some_and(|a| a.iter().any(|s| s == "bound-symbols-resolve")),
        "{}",
        show(&reported)
    );

    let armed = gate("code-ghost");
    assert_eq!(armed.code, 1, "{}", show(&armed));
    let va = json_of(&armed);
    assert_eq!(va["verdict"], "Fail", "{}", show(&armed));
    assert_eq!(va["violations"], 1, "{}", show(&armed));
    let findings = va["findings"].as_array().expect("findings").clone();
    assert!(
        findings
            .iter()
            .any(|f| f["message"].as_str().is_some_and(|m| m
                .contains("symbol/kern::nn::functional::no_such_function")
                && m.contains("bound-symbols-resolve"))),
        "the violation names the symbol and the shape: {}",
        show(&armed)
    );
}

#[test]
fn a_theorem_file_carrying_sorry_grounds_nothing_and_the_shape_says_so() {
    let r = gate("lean-sorry");
    assert_eq!(r.code, 1, "{}", show(&r));
    let v = json_of(&r);
    assert_eq!(v["lean_statements"], 2, "{}", show(&r));
    assert_eq!(v["by_entity_type"]["lean"], 2, "{}", show(&r));
    let findings = v["findings"].as_array().expect("findings").clone();
    assert!(
        findings
            .iter()
            .any(|f| f["message"].as_str().is_some_and(|m| {
                m.contains("world/Softmax.Core.softmax_sums_to_one")
                    && m.contains("lean-statements-grounded")
            })),
        "the violation names the admitted theorem: {}",
        show(&r)
    );
}

#[test]
fn a_lean_reference_naming_no_theorem_is_counted_not_guessed() {
    let r = gate("lean-sorry");
    let v = json_of(&r);
    // `Theorems.NothingOfTheSort` resolves to nothing in the fixture's tree.
    assert_eq!(v["lean_refs_unresolved"], 1, "{}", show(&r));
}

#[test]
fn the_vendored_w3c_cases_run_every_gate_run_and_all_pass() {
    let r = gate("code-bound");
    let v = json_of(&r);
    let n = v["w3c_cases_n"].as_u64().expect("w3c_cases_n");
    let passed = v["w3c_cases_passed"].as_u64().expect("w3c_cases_passed");
    assert!(n >= 16, "at least the 16 vendored cases: {}", show(&r));
    assert_eq!(passed, n, "every vendored case passes: {}", show(&r));
}

#[test]
fn extract_writes_symbol_and_world_nodes_with_no_blank_node() {
    let dir = fixture("code-bound");
    let r = pv_in(&dir, &["extract", "contracts"]);
    assert_eq!(r.code, 0, "{}", show(&r));
    let nt = std::fs::read_to_string(dir.join("contracts/contracts.nt")).expect("contracts.nt");
    assert!(
        nt.contains("/symbol/kern::nn::functional::softmax>"),
        "{nt}"
    );
    assert!(nt.contains("/sym/attribute> \"kernel\""), "{nt}");
    assert!(!nt.contains("_:"), "no blank node");
    // Written by the same walk the gate grades (R-18): a second extraction is byte-identical.
    let again = pv_in(&dir, &["extract", "contracts"]);
    assert_eq!(again.code, 0, "{}", show(&again));
    let nt2 = std::fs::read_to_string(dir.join("contracts/contracts.nt")).expect("contracts.nt");
    assert_eq!(nt, nt2);
    // The fixture's tracked graph is not part of the fixture: leave the tree as it was found.
    let _ = std::fs::remove_file(dir.join("contracts/contracts.nt"));
    let _ = std::fs::remove_file(dir.join("contracts/shapes.ttl"));
}

#[test]
fn the_lean_tree_is_found_under_the_repo_root_never_the_process_cwd() {
    // Run from a scratch cwd with an ABSOLUTE contract dir: the Lean tree must still be the fixture's.
    let scratch = tempfile::tempdir().expect("scratch");
    let dir = fixture("lean-sorry").join("contracts");
    let r = pv_in(
        scratch.path(),
        &[
            "lint",
            dir.to_str().expect("utf-8"),
            "--gate",
            "shapes",
            "--format",
            "json",
        ],
    );
    let v = json_of(&r);
    assert_eq!(v["lean_statements"], 2, "{}", show(&r));
}

#[test]
fn the_oracle_crate_is_outside_the_workspace_and_no_shacl_crate_is_reachable() {
    // R-13 / F-20, the falsifier ONT-0's ledger states: the pinned oracle may never enter the gate path.
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let oracle =
        std::fs::read_to_string(root.join("tests/oracle/Cargo.toml")).expect("oracle manifest");
    assert!(
        oracle.contains("[workspace]"),
        "the oracle crate detaches itself from the workspace"
    );
    assert!(
        oracle.contains("shacl = { version = \"=0.3.21\""),
        "{oracle}"
    );
    let lock = std::fs::read_to_string(root.join("Cargo.lock")).expect("workspace lock");
    for forbidden in [
        "name = \"shacl\"",
        "name = \"rudof_rdf\"",
        "name = \"oxigraph\"",
    ] {
        assert!(
            !lock.contains(forbidden),
            "{forbidden} is in the WORKSPACE lock — the oracle leaked into the gate path (R-13)"
        );
    }
}
