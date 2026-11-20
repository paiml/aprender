//! #7245 / PMAT-7245 (SIM3D P0) — a simulation spec and receipt under contract
//! (`contracts/sim3d-spec-v1.yaml`, `contracts/sim3d-receipt-v1.yaml`), graded by pv on the CLI.
//!
//! Each `.sim.yaml` fixture is emitted as canonical JSON (sorted keys, no whitespace), copied into a scratch
//! corpus beside the REAL contract with an `entity:` block appended, and judged with
//! `pv lint <scratch>/contracts --gate shapes --shape <id>`. The fixtures carry no copy of the contract, so
//! widening a shape there turns this table red. Every refused fixture states the path and the component pv must
//! name; a refusal for a different reason does not satisfy the row.
//!
//! This target IS wired into `ci/explicit-test-commands.d/` (fragment 871) and runs on every pull request, so it
//! is a GATE check; wiring it was a gate change (`GATE CHANGE: yes`). Only the invariant-map checker
//! `scripts/lab/check_sim3d_spec_contracts.sh` is LAB (nightly or by hand).

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

fn pv_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pv"))
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture_path(name: &str) -> PathBuf {
    repo_root().join("tests/fixtures/ont/sim3d").join(name)
}

/// Sorted-key rebuild: canonical whatever order the YAML was written in.
fn canonical(v: &Value) -> Value {
    match v {
        Value::Object(m) => {
            let sorted: std::collections::BTreeMap<_, _> =
                m.iter().map(|(k, x)| (k.clone(), canonical(x))).collect();
            Value::Object(sorted.into_iter().collect())
        }
        Value::Array(a) => Value::Array(a.iter().map(canonical).collect()),
        other => other.clone(),
    }
}

/// The fixture as canonical JSON text: a `.sim.yaml` is converted, a `.json` is re-emitted.
fn canonical_json(name: &str) -> String {
    let text = std::fs::read_to_string(fixture_path(name))
        .unwrap_or_else(|e| panic!("fixture {name} unreadable: {e}"));
    let v: Value = if Path::new(name)
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("json"))
    {
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("{name} is not JSON: {e}"))
    } else {
        serde_saphyr::from_str(&text).unwrap_or_else(|e| panic!("{name} is not YAML: {e}"))
    };
    serde_json::to_string(&canonical(&v)).expect("serialises")
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

    fn report(&self) -> Value {
        serde_json::from_str(&self.stdout)
            .unwrap_or_else(|e| panic!("pv's report is JSON ({e})\n{}", self.all()))
    }

    /// Every Error-severity `message` in the report.
    fn errors(&self) -> Vec<String> {
        fn walk(v: &Value, out: &mut Vec<String>) {
            match v {
                Value::Object(m) => {
                    if let (Some(msg), Some("Error")) = (
                        m.get("message").and_then(Value::as_str),
                        m.get("severity").and_then(Value::as_str),
                    ) {
                        out.push(msg.to_owned());
                    }
                    m.values().for_each(|x| walk(x, out));
                }
                Value::Array(a) => a.iter().for_each(|x| walk(x, out)),
                _ => {}
            }
        }
        let mut out = Vec::new();
        walk(&self.report(), &mut out);
        out
    }
}

/// Judge `fixture` on the real contract `contract_file` (shape id = its stem).
fn judge(contract_file: &str, fixture: &str) -> Run {
    judge_text(contract_file, &canonical_json(fixture))
}

/// Judge an in-memory document (a mutated fixture) the same way.
fn judge_value(contract_file: &str, doc: &Value) -> Run {
    judge_text(
        contract_file,
        &serde_json::to_string(&canonical(doc)).expect("serialises"),
    )
}

fn judge_text(contract_file: &str, doc: &str) -> Run {
    let shape = contract_file.trim_end_matches(".yaml");
    let contract = std::fs::read_to_string(repo_root().join("contracts").join(contract_file))
        .expect("contract readable");
    assert!(
        !contract.contains("\nentity:"),
        "{contract_file} must not declare an entity: this harness appends one"
    );
    let dir = tempfile::tempdir().expect("scratch");
    std::fs::create_dir(dir.path().join("contracts")).expect("mkdir");
    std::fs::create_dir(dir.path().join("data")).expect("mkdir");
    std::fs::write(
        dir.path().join("contracts").join(contract_file),
        format!("{contract}\nentity:\n  type: json\n  ref: data/doc.json\n"),
    )
    .expect("write contract");
    std::fs::write(dir.path().join("data/doc.json"), doc).expect("write doc");
    let out = Command::new(pv_bin())
        .args(["lint"])
        .arg(dir.path().join("contracts"))
        .args(["--gate", "shapes", "--shape", shape])
        .output()
        .expect("spawn pv");
    Run {
        code: out.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

fn fixture_value(name: &str) -> Value {
    serde_json::from_str(&canonical_json(name)).expect("canonical json parses")
}

const SPEC: &str = "sim3d-spec-v1.yaml";
const RECEIPT: &str = "sim3d-receipt-v1.yaml";

fn assert_pass(contract: &str, fixture: &str) {
    let r = judge(contract, fixture);
    assert_eq!(r.code, 0, "{fixture} must conform\n{}", r.all());
    let v = r.report();
    assert_eq!(v["verdict"].as_str(), Some("Pass"), "{}", r.all());
    assert_eq!(v["extra"]["violations"].as_u64(), Some(0), "{}", r.all());
    assert_eq!(
        v["extra"]["focus_nodes_n"].as_u64(),
        Some(1),
        "{fixture}: exactly one focus node, or the pass is vacuous\n{}",
        r.all()
    );
    assert!(
        v["extra"]["armed_shapes"]
            .as_array()
            .expect("armed_shapes")
            .iter()
            .any(|s| s.as_str() == Some(contract.trim_end_matches(".yaml"))),
        "the harness must arm the shape\n{}",
        r.all()
    );
}

/// `(fixture, component, path fragment)`: pv must exit 1 and one Error must carry both.
fn assert_refused(contract: &str, rows: &[(&str, &str, &str)]) {
    for (fixture, component, path) in rows {
        let r = judge(contract, fixture);
        assert_eq!(r.code, 1, "{fixture} was not refused\n{}", r.all());
        let errors = r.errors();
        assert!(
            errors
                .iter()
                .any(|m| m.contains(&format!("({component})")) && m.contains(path)),
            "{fixture}: expected a ({component}) violation naming `{path}`, got {errors:#?}\n{}",
            r.all()
        );
    }
}

#[test]
fn the_mm1_experiment_and_the_load_balancer_specs_conform() {
    assert_pass(SPEC, "mm1-littles-law.sim.yaml");
    assert_pass(SPEC, "lb-round-robin.sim.yaml");
}

#[test]
fn a_law_naming_an_undeclared_invariant_is_refused_by_pv() {
    // pv itself refuses it: the extractor resolves the id against `invariants[].id`, the shape asks for the class.
    assert_refused(
        SPEC,
        &[(
            "invalid-law-undeclared-invariant.sim.yaml",
            "node",
            "invariants.undeclared.INV-999> is not an instance of ont:sim/Invariant",
        )],
    );
    // declaring the id clears it: the same spec with the invariant added conforms
    let mut spec = fixture_value("invalid-law-undeclared-invariant.sim.yaml");
    let mut extra = spec["invariants"][0].clone();
    extra["id"] = serde_json::json!("INV-999");
    spec["invariants"]
        .as_array_mut()
        .expect("invariants")
        .push(extra);
    let r = judge_value(SPEC, &spec);
    assert_eq!(r.code, 0, "a declared id must conform\n{}", r.all());
    // a second dangling law is refused too: remove INV-999 and point two laws at nothing
    let mut two = spec.clone();
    two["invariants"].as_array_mut().expect("invariants").pop();
    let laws = two["laws"].as_array_mut().expect("laws");
    let mut again = laws[0].clone();
    again["id"] = serde_json::json!("LAW-998");
    again["invariant"] = serde_json::json!("INV-998");
    laws.push(again);
    let r = judge_value(SPEC, &two);
    assert_eq!(r.code, 1, "two dangling laws\n{}", r.all());
    let n = r
        .errors()
        .iter()
        .filter(|m| m.contains("is not an instance of ont:sim/Invariant"))
        .count();
    assert!(n >= 2, "both dangling laws are named, got {n}\n{}", r.all());
}

#[test]
fn the_planted_falsifiers_each_go_red_naming_the_path() {
    assert_refused(
        SPEC,
        &[
            // the three falsifiers of the phase-0 verification line
            ("invalid-missing-seed.sim.yaml", "minCount", "ont:sim/seed"),
            (
                "invalid-no-invariants.sim.yaml",
                "minCount",
                "ont:sim/invariants",
            ),
            (
                "invalid-empty-invariants.sim.yaml",
                "minCount",
                "ont:sim/invariants",
            ),
            ("invalid-empty-spec.sim.yaml", "minCount", "ont:sim/seed"),
            (
                "invalid-empty-spec.sim.yaml",
                "minCount",
                "ont:sim/invariants",
            ),
            ("invalid-empty-spec.sim.yaml", "minCount", "ont:sim/laws"),
            (
                "invalid-empty-spec.sim.yaml",
                "minCount",
                "ont:sim/entities",
            ),
        ],
    );
}

#[test]
fn the_closed_sets_and_shapes_refuse_each_for_its_own_reason() {
    assert_refused(
        SPEC,
        &[
            ("invalid-extra-key.sim.yaml", "closed", "ieee_strict"),
            ("invalid-bad-unit.sim.yaml", "node", "ont:sim/unit"),
            (
                "invalid-bad-seed-algorithm.sim.yaml",
                "node",
                "ont:sim/algorithm",
            ),
            ("invalid-seed-no-value.sim.yaml", "node", "ont:sim/value"),
            ("invalid-bad-entity-kind.sim.yaml", "node", "ont:sim/kind"),
            (
                "invalid-no-entities.sim.yaml",
                "minCount",
                "ont:sim/entities",
            ),
            (
                "invalid-law-without-invariant.sim.yaml",
                "node",
                "ont:sim/invariant",
            ),
            (
                "invalid-law-bad-invariant-id.sim.yaml",
                "node",
                "is not an instance of ont:sim/Invariant",
            ),
            (
                "invalid-zero-horizon.sim.yaml",
                "pattern",
                "ont:sim/horizon_ns",
            ),
            ("invalid-bad-schema-tag.sim.yaml", "in", "ont:sim/schema"),
            ("invalid-float-clock.sim.yaml", "in", "ont:sim/time_unit"),
        ],
    );
}

#[test]
fn the_receipt_shape_has_both_arms() {
    assert_pass(RECEIPT, "receipt-mm1.json");
    assert_refused(
        RECEIPT,
        &[
            (
                "invalid-receipt-no-final-hash.json",
                "minCount",
                "ont:sim/final_hash",
            ),
            (
                "invalid-receipt-short-hash.json",
                "pattern",
                "ont:sim/final_hash",
            ),
            (
                "invalid-receipt-no-verdicts.json",
                "minCount",
                "ont:sim/verdicts",
            ),
            ("invalid-receipt-extra-key.json", "closed", "host"),
            (
                "invalid-receipt-empty.json",
                "minCount",
                "ont:sim/final_hash",
            ),
        ],
    );
}

#[test]
fn every_fixture_in_the_table_is_judged_and_none_is_orphaned() {
    // A fixture no row names is dark: it would read as covered and grade nothing.
    let named: Vec<&str> = vec![
        "mm1-littles-law.sim.yaml",
        "lb-round-robin.sim.yaml",
        "receipt-mm1.json",
        "invalid-missing-seed.sim.yaml",
        "invalid-no-invariants.sim.yaml",
        "invalid-empty-invariants.sim.yaml",
        "invalid-empty-spec.sim.yaml",
        "invalid-extra-key.sim.yaml",
        "invalid-bad-unit.sim.yaml",
        "invalid-bad-seed-algorithm.sim.yaml",
        "invalid-seed-no-value.sim.yaml",
        "invalid-bad-entity-kind.sim.yaml",
        "invalid-no-entities.sim.yaml",
        "invalid-law-without-invariant.sim.yaml",
        "invalid-law-bad-invariant-id.sim.yaml",
        "invalid-law-undeclared-invariant.sim.yaml",
        "invalid-zero-horizon.sim.yaml",
        "invalid-bad-schema-tag.sim.yaml",
        "invalid-float-clock.sim.yaml",
        "invalid-receipt-no-final-hash.json",
        "invalid-receipt-short-hash.json",
        "invalid-receipt-no-verdicts.json",
        "invalid-receipt-extra-key.json",
        "invalid-receipt-empty.json",
    ];
    let mut on_disk: Vec<String> = std::fs::read_dir(repo_root().join("tests/fixtures/ont/sim3d"))
        .expect("fixtures dir")
        .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
        .collect();
    on_disk.sort();
    let mut want: Vec<String> = named.iter().map(|s| (*s).to_owned()).collect();
    want.sort();
    assert_eq!(
        on_disk, want,
        "fixtures on disk and fixtures in the table differ"
    );
}
