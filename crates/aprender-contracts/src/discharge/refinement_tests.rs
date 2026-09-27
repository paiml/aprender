//! ONT-001 ONT-3b (#4073) — what a `models:` entry must say, and which modules earn L4 from it.

use super::*;
use crate::discharge::summary::{discharged, Module, Summary};

const SM: &str = "ProvableContracts/Theorems/Softmax/Kernel.lean";
const GELU: &str = "ProvableContracts/Theorems/Gelu/Bound.lean";

fn rec(yaml: &str) -> Record {
    parse(&serde_yaml::from_str(yaml).expect("yaml"))
}

fn model(module: &str, model_of: &str, kind: Kind) -> Model {
    Model {
        module: module.into(),
        model_of: model_of.into(),
        kind,
        evidence: "e".into(),
    }
}

fn tree() -> BTreeSet<String> {
    [SM, GELU].iter().map(|s| (*s).to_string()).collect()
}

/// Resolves only `trueno::softmax::softmax_row`.
fn resolver(p: &str) -> Result<(), String> {
    if p == "trueno::softmax::softmax_row" {
        Ok(())
    } else {
        Err(format!("no item {p}"))
    }
}

#[test]
fn only_extraction_and_simulation_reach_l4() {
    assert_eq!(Kind::parse("extraction").map(Kind::level), Some(4));
    assert_eq!(Kind::parse("simulation").map(Kind::level), Some(4));
    assert_eq!(Kind::parse("test_witnessed").map(Kind::level), Some(3));
    assert_eq!(Kind::parse("proved"), None);
    for k in [Kind::Extraction, Kind::Simulation, Kind::TestWitnessed] {
        assert_eq!(Kind::parse(k.as_str()), Some(k));
    }
}

#[test]
fn a_well_formed_record_parses_and_every_malformed_entry_is_named() {
    let r = rec(r#"
models:
  - { module: "a.lean", model_of: "c::m::f", relation: { kind: extraction, evidence: "aeneas" } }
  - { module: "b.lean", relation: { kind: extraction, evidence: "x" } }
  - { module: "c.lean", model_of: "c::m::f", relation: { kind: proved, evidence: "x" } }
  - { module: "d.lean", model_of: "nopath", relation: { kind: simulation, evidence: "x" } }
  - { module: "e.lean", model_of: "c::m::f", relation: { kind: simulation, evidence: "" } }
  - { module: "a.lean", model_of: "c::m::g", relation: { kind: simulation, evidence: "x" } }
unrefined_baseline: 7
"#);
    assert_eq!(
        r.models,
        vec![Model {
            evidence: "aeneas".into(),
            ..model("a.lean", "c::m::f", Kind::Extraction)
        }]
    );
    assert_eq!(r.unrefined_baseline, Some(7));
    assert_eq!(r.malformed.len(), 5, "{:?}", r.malformed);
    assert!(r.malformed[4].contains("modelled twice"));
    assert_eq!(rec("{}"), Record::default(), "no models, no baseline");
}

#[test]
fn a_model_grants_its_level_only_when_it_resolves_and_names_a_tree_module() {
    let j = judge(
        &[
            model(SM, "trueno::softmax::softmax_row", Kind::Extraction),
            model(GELU, "trueno::gelu::ghost", Kind::Extraction),
            model(
                "Not/In/Tree.lean",
                "trueno::softmax::softmax_row",
                Kind::Simulation,
            ),
        ],
        &tree(),
        resolver,
    );
    assert_eq!(j[0].level(), Some(4));
    assert_eq!(j[1].level(), None, "a ghost model_of grants nothing");
    assert!(j[1].resolved.as_ref().is_err_and(|w| w.contains("ghost")));
    assert_eq!(
        j[2].level(),
        None,
        "a module outside the tree grants nothing"
    );
    assert_eq!(l4_modules(&j), BTreeSet::from([SM.to_string()]));

    let tw = judge(
        &[model(
            SM,
            "trueno::softmax::softmax_row",
            Kind::TestWitnessed,
        )],
        &tree(),
        resolver,
    );
    assert_eq!(tw[0].level(), Some(3));
    assert!(l4_modules(&tw).is_empty(), "test_witnessed is L3, never L4");
}

#[test]
fn unrefined_counts_theorem_bearing_modules_without_an_l4_model() {
    let tm: BTreeMap<String, Vec<String>> = [
        (SM.to_string(), vec!["T.sm".to_string()]),
        (GELU.to_string(), vec!["T.gelu".to_string()]),
        ("Empty.lean".to_string(), vec![]),
    ]
    .into_iter()
    .collect();
    assert_eq!(unrefined(&tm, &BTreeSet::new()), vec![GELU, SM]);
    assert_eq!(
        unrefined(&tm, &BTreeSet::from([SM.to_string()])),
        vec![GELU]
    );
}

fn green() -> Summary {
    let m = |p: &str, t: &str| Module {
        path: p.into(),
        blake3: "b".into(),
        theorems: vec![t.into()],
    };
    Summary {
        tree_sha: Some("t".into()),
        build_exit: Some(0),
        lake_exit: Some(0),
        leanchecker_exit: Some(0),
        axioms_ok: true,
        escapes_ok: true,
        challenges_closed: Some("1/1".into()),
        modules: vec![m(SM, "T.sm"), m(GELU, "T.gelu")],
        ..Summary::default()
    }
}

/// The spec's mutation: delete one `model_of` and that module's L4 credit disappears; the other keeps its own.
#[test]
fn deleting_a_model_of_removes_exactly_that_modules_credit() {
    let both = [
        model(SM, "trueno::softmax::softmax_row", Kind::Extraction),
        model(GELU, "trueno::softmax::softmax_row", Kind::Simulation),
    ];
    let credit = |models: &[Model]| {
        let mut g = discharged(&green(), Some("t"));
        g.require_refinement(&l4_modules(&judge(models, &tree(), resolver)));
        g
    };
    let g = credit(&both);
    assert!(g.grants("T.sm") && g.grants("T.gelu"));
    assert!(g.unrefined.is_empty());

    let g = credit(&both[..1]);
    assert!(g.grants("T.sm"));
    assert!(!g.grants("T.gelu"), "GELU lost its model, so its credit");
    assert_eq!(g.unrefined, BTreeSet::from(["T.gelu".to_string()]));

    let g = credit(&[]);
    assert!(
        g.theorems.is_empty(),
        "discharged with no model_of is not L4"
    );
}
