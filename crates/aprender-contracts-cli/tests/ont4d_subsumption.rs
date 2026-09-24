//! ONT-4d (PMAT-4070, aprender#4070): subsumption in Σ; shapes inherit down the hierarchy. Every RED clause of
//! the row, driven through the BUILT `pv`.
//!
//! - cycle → exit 3 `error: subsumes cycle <path>`                 `subsumption-cycle/`
//! - a shape on `Code` rejects a `Kernel` instance; focus_nodes_n == instances of Code ∪ all sub-concepts
//!                                                                  `subsumption-inherit/`
//! - a sub-concept shape removing a super constraint → exit 1 `reject: <sub> weakens <super>.<constraint>`
//!                                                                  `subsumption-weaken/`
//! - rdf:type closure present in contracts.nt for every super-concept (tracked corpus, and a fixture extract)
//! - export byte-identical across two runs
//!
//! DISCRIMINATION: `subsumption-ok/` (the same hierarchy, a valid kernel) is Pass at exit 0, so a build that
//! rejects every inherited instance fails this file. The row's MUTATION (drop the closure materialization) turns
//! `a_code_shape_rejects_a_kernel_instance_through_the_closure` RED: with no closure, the Code shape has no
//! focus node, and the gate declines instead of rejecting.

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

fn pv(args: &[&str]) -> Run {
    let scratch = tempfile::tempdir().expect("scratch cwd is creatable");
    let out = Command::new(pv_bin())
        .current_dir(scratch.path())
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

fn repo(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(rel)
}

fn fixture(name: &str) -> String {
    s(&repo(&format!("tests/fixtures/ont/{name}")))
}

fn s(p: &Path) -> String {
    p.to_str().expect("utf-8 path").to_string()
}

fn shapes_json(dir: &str) -> (Run, serde_json::Value) {
    let r = pv(&["lint", dir, "--gate", "shapes", "--format", "json"]);
    let v = serde_json::from_str(&r.stdout)
        .unwrap_or_else(|e| panic!("stdout is JSON: {e}\n{}", show(&r)));
    (r, v)
}

#[test]
fn a_subsumption_cycle_is_exit_3_naming_the_path() {
    let r = pv(&["lint", &fixture("subsumption-cycle"), "--gate", "sigma"]);
    assert_eq!(r.code, 3, "{}", show(&r));
    assert!(r.stderr.contains("error: subsumes cycle "), "{}", show(&r));
    for c in ["Kernel", "Code", "Contract"] {
        assert!(
            r.stderr.contains(c),
            "the cycle path names {c}: {}",
            show(&r)
        );
    }
}

#[test]
fn a_code_shape_rejects_a_kernel_instance_through_the_closure() {
    let (r, v) = shapes_json(&fixture("subsumption-inherit"));
    assert_eq!(r.code, 1, "{}", show(&r));
    assert_eq!(v["verdict"], "Fail", "{}", show(&r));
    // focus_nodes_n == instances of Code ∪ all sub-concepts: no contract is typed Code directly; the one kernel
    // contract reaches the Code shape ONLY through Kernel ⊑ Code.
    assert_eq!(v["focus_nodes_n"], 1, "{}", show(&r));
    assert_eq!(
        v["by_shape"],
        serde_json::Value::from(vec!["code-shape=1"]),
        "{}",
        show(&r)
    );
    assert_eq!(v["inherited_shapes_applied"], 1, "{}", show(&r));
    assert_eq!(v["violations"], 1, "{}", show(&r));
}

#[test]
fn the_same_hierarchy_with_a_valid_kernel_passes() {
    let (r, v) = shapes_json(&fixture("subsumption-ok"));
    assert_eq!(r.code, 0, "{}", show(&r));
    assert_eq!(v["verdict"], "Pass", "{}", show(&r));
    assert_eq!(v["inherited_shapes_applied"], 1, "{}", show(&r));
    assert_eq!(
        v["inherited_by_shape"],
        serde_json::Value::from(vec!["code-shape <- Kernel=1"]),
        "{}",
        show(&r)
    );
}

#[test]
fn a_sub_shape_that_weakens_a_super_shape_is_rejected() {
    let (r, v) = shapes_json(&fixture("subsumption-weaken"));
    assert_eq!(r.code, 1, "{}", show(&r));
    let msgs: Vec<String> = v["findings"]
        .as_array()
        .expect("findings")
        .iter()
        .map(|f| f["message"].as_str().unwrap_or_default().to_string())
        .collect();
    assert!(
        msgs.iter()
            .any(|m| m.starts_with("reject: kernel-shape weakens contract-shape.name.minCount")),
        "{msgs:?}\n{}",
        show(&r)
    );
    // The kernel instance satisfies BOTH shapes, so the weakening is the ONLY violation.
    assert_eq!(v["violations"], 1, "{}", show(&r));
}

#[test]
fn the_repo_corpus_passes_with_shapes_applied_down_the_hierarchy() {
    // The row's probe.
    let sigma = std::fs::read_to_string(repo("contracts/ontology.yaml")).expect("Σ");
    assert!(
        sigma.lines().any(|l| l.starts_with("subsumes:")),
        "Σ declares subsumes"
    );
    let (r, v) = shapes_json(&s(&repo("contracts")));
    assert_eq!(v["verdict"], "Pass", "{}", show(&r));
    assert!(
        v["inherited_shapes_applied"].as_u64().unwrap_or(0) > 0,
        "{}",
        show(&r)
    );
    let ofn = std::fs::read_to_string(repo("contracts/ontology.ofn")).expect("ontology.ofn");
    assert!(ofn.contains("SubClassOf(<https://ont.paiml.dev/v1alpha1/Kernel> <https://ont.paiml.dev/v1alpha1/Contract>)"));
    assert!(ofn.contains(
        "SubClassOf(<https://ont.paiml.dev/v1alpha1/Symbol> <https://ont.paiml.dev/v1alpha1/Code>)"
    ));
}

/// Subjects typed `class` in an N-Triples text.
fn typed(nt: &str, class: &str) -> std::collections::BTreeSet<String> {
    let needle = format!("> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <https://ont.paiml.dev/v1alpha1/{class}> .");
    nt.lines()
        .filter(|l| l.ends_with(&needle))
        .map(|l| l.split(' ').next().unwrap_or_default().to_string())
        .collect()
}

#[test]
fn the_tracked_contracts_nt_carries_the_type_closure() {
    let nt = std::fs::read_to_string(repo("contracts/contracts.nt")).expect("contracts.nt");
    let kernels = typed(&nt, "Kernel");
    let symbols = typed(&nt, "Symbol");
    assert!(
        !kernels.is_empty() && !symbols.is_empty(),
        "both sub-concepts have instances"
    );
    assert!(
        kernels.is_subset(&typed(&nt, "Contract")),
        "every Kernel is typed Contract"
    );
    assert!(
        symbols.is_subset(&typed(&nt, "Code")),
        "every Symbol is typed Code (closure-only: no extractor types Code)"
    );
}

#[test]
fn a_fixture_extract_carries_the_closure_and_is_byte_identical_twice() {
    let d = tempfile::tempdir().expect("tempdir");
    for f in ["ontology.yaml", "code-shape.yaml", "kernel-noname.yaml"] {
        std::fs::copy(
            repo(&format!("tests/fixtures/ont/subsumption-inherit/{f}")),
            d.path().join(f),
        )
        .expect("copy");
    }
    let dir = s(d.path());
    let r = pv(&["extract", &dir]);
    assert_eq!(r.code, 0, "{}", show(&r));
    let first = std::fs::read(d.path().join("contracts.nt")).expect("contracts.nt written");
    let r = pv(&["extract", &dir]);
    assert_eq!(r.code, 0, "{}", show(&r));
    let second = std::fs::read(d.path().join("contracts.nt")).expect("contracts.nt written");
    assert_eq!(first, second, "export byte-identical across two runs");
    let nt = String::from_utf8(first).expect("utf-8");
    let k = typed(&nt, "Kernel");
    assert_eq!(k.len(), 1, "{nt}");
    assert!(
        k.is_subset(&typed(&nt, "Code")) && k.is_subset(&typed(&nt, "Contract")),
        "closure over two levels: {nt}"
    );
}

#[test]
fn census_reports_by_concept_over_the_closure() {
    let r = pv(&["census", &s(&repo("contracts")), "--format", "json"]);
    assert_eq!(r.code, 0, "{}", show(&r));
    let v: serde_json::Value = serde_json::from_str(&r.stdout).expect("census JSON");
    let n = |c: &str| v["by_concept"][c].as_u64().unwrap_or(0);
    assert!(n("Kernel") > 0 && n("Symbol") > 0, "{}", show(&r));
    assert!(
        n("Contract") >= n("Kernel"),
        "Contract counts its Kernel sub-instances"
    );
    assert!(
        n("Code") >= n("Symbol"),
        "Code counts its Symbol sub-instances"
    );
}
