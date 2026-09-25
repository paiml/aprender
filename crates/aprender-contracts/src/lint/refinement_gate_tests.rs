//! ONT-001 ONT-3b (#4073) — the refinement gate: ghosts, malformed entries, and the shrink-only baseline.

use super::*;
use crate::discharge::summary::{Module, Summary};

const SM: &str = "ProvableContracts/Theorems/Softmax/Kernel.lean";
const GELU: &str = "ProvableContracts/Theorems/Gelu/Bound.lean";

/// A lean dir (`<tmp>/lean`) with a summary of two theorem-bearing modules beside it and `formalization.yaml`.
fn fixture(formalization: &str) -> tempfile::TempDir {
    let d = tempfile::tempdir().expect("tempdir");
    let lean = d.path().join("lean");
    std::fs::create_dir(&lean).expect("mkdir");
    let m = |p: &str, t: &str| Module {
        path: p.into(),
        blake3: "b".into(),
        theorems: vec![t.into()],
    };
    let s = Summary {
        modules: vec![m(SM, "T.sm"), m(GELU, "T.gelu")],
        ..Summary::default()
    };
    std::fs::write(summary::summary_path(&lean), s.render()).expect("summary");
    std::fs::write(lean.join("formalization.yaml"), formalization).expect("formalization");
    d
}

fn resolver(p: &str) -> Result<(), String> {
    if p.ends_with("::real") {
        Ok(())
    } else {
        Err(format!("no item {p}"))
    }
}

fn run(formalization: &str) -> (Verdict, Vec<String>, RefinementCounters) {
    let d = fixture(formalization);
    match run_with(&d.path().join("lean"), resolver) {
        RatchetOutcome::Ran { result, findings } => {
            let Some(GateExtra::Refinement(c)) = result.extra else {
                panic!("no counters")
            };
            (
                result.verdict,
                findings.iter().map(|f| f.rule_id.clone()).collect(),
                *c,
            )
        }
        RatchetOutcome::Declined(why) => panic!("declined: {why}"),
    }
}

fn models(kind: &str, a: &str) -> String {
    format!(
        "models:\n  - {{ module: {SM}, model_of: c::m::{a}, relation: {{ kind: {kind}, evidence: e }} }}\n"
    )
}

#[test]
fn a_resolving_model_and_an_exact_baseline_pass() {
    let (v, f, c) = run(&(models("extraction", "real") + "unrefined_baseline: 1\n"));
    assert_eq!((v, f.len()), (Verdict::Pass, 0), "{f:?}");
    assert_eq!(
        (
            c.models,
            c.resolved,
            c.l4_models,
            c.unrefined,
            c.theorem_modules
        ),
        (1, 1, 1, 1, 2)
    );

    let (v, _, c) = run(&(models("test_witnessed", "real") + "unrefined_baseline: 2\n"));
    assert_eq!(v, Verdict::Pass);
    assert_eq!(
        (c.l3_models, c.l4_models, c.unrefined),
        (1, 0, 2),
        "test_witnessed refines nothing"
    );
}

#[test]
fn a_ghost_model_of_is_named_and_refines_nothing() {
    let (v, f, c) = run(&(models("extraction", "ghost") + "unrefined_baseline: 2\n"));
    assert_eq!(v, Verdict::Fail);
    assert_eq!(f, vec!["PV-ONT-031"]);
    assert_eq!((c.ghosts, c.l4_models, c.unrefined), (1, 0, 2));
}

#[test]
fn a_malformed_entry_or_an_unknown_module_is_032() {
    let (_, f, _) = run("models:\n  - { module: X.lean, model_of: c::m::real, relation: { kind: extraction, evidence: e } }\nunrefined_baseline: 2\n");
    assert_eq!(f, vec!["PV-ONT-032"]);
    let (_, f, _) = run("models:\n  - { module: X.lean, relation: { kind: extraction, evidence: e } }\nunrefined_baseline: 2\n");
    assert_eq!(f, vec!["PV-ONT-032"]);
}

#[test]
fn the_baseline_is_required_and_shrink_only() {
    for (baseline, why) in [
        ("", "absent"),
        ("unrefined_baseline: 1\n", "below"),
        ("unrefined_baseline: 3\n", "stale above"),
    ] {
        let (v, f, _) = run(&format!("models: []\n{baseline}"));
        assert_eq!(
            (v, f),
            (Verdict::Fail, vec!["PV-ONT-033".to_string()]),
            "{why}"
        );
    }
    assert_eq!(run("models: []\nunrefined_baseline: 2\n").0, Verdict::Pass);
}

#[test]
fn no_summary_or_no_formalization_declines() {
    let d = tempfile::tempdir().expect("tempdir");
    assert!(matches!(
        run_with(&d.path().join("lean"), resolver),
        RatchetOutcome::Declined(_)
    ));
    let d = fixture("");
    std::fs::remove_file(d.path().join("lean/formalization.yaml")).expect("rm");
    assert!(matches!(
        run_with(&d.path().join("lean"), resolver),
        RatchetOutcome::Declined(_)
    ));
}
