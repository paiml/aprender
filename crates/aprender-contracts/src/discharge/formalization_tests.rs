//! EV-8b `--validate-formalization`: a green record, then each measured field varied alone.

use super::*;
use crate::discharge::summary::Module;

const GOOD: &str = r"
main_results:
  - ProvableContracts.Gelu.gelu_bound
status:
  axioms: [propext, Classical.choice, Quot.sound]
capstones: []
sorry_count: 2
scope: the Rust binding is not covered
review:
  status: self-assessed
automation:
  methods: [simp]
";

fn summary() -> Summary {
    Summary {
        modules: vec![Module {
            path: "ProvableContracts/Theorems/Gelu.lean".into(),
            blake3: "0".repeat(64),
            theorems: vec!["ProvableContracts.Gelu.gelu_bound".into()],
        }],
        ..Summary::default()
    }
}

fn judged(yaml: &str, s: Option<&Summary>, sorry: usize) -> Report {
    let doc: Value = serde_yaml::from_str(yaml).unwrap();
    let mut r = Report::default();
    judge_doc(&doc, s, sorry, &mut r);
    r
}

#[test]
fn a_record_matching_the_measurements_passes() {
    let r = judged(GOOD, Some(&summary()), 2);
    assert!(!r.reject, "{:?}", r.lines);
    assert!(r.lines[0].starts_with("ok"), "{:?}", r.lines);
}

#[test]
fn each_measured_field_that_disagrees_is_red_by_name() {
    let cases = [
        (
            GOOD.replace("Gelu.gelu_bound\n", "Gelu.not_discharged\n"),
            2,
            "main_results ProvableContracts.Gelu.not_discharged",
        ),
        (
            GOOD.replace(", Quot.sound]", ", Quot.sound, Foo.ax]"),
            2,
            "status.axioms",
        ),
        (
            GOOD.replace("[propext, Classical.choice, Quot.sound]", "[propext]"),
            2,
            "status.axioms",
        ),
        (GOOD.to_string(), 3, "sorry_count 2 != 3"),
        (
            GOOD.replace("sorry_count: 2\n", ""),
            2,
            "sorry_count is missing",
        ),
        (
            GOOD.replace("  status: self-assessed\n", "  status: ''\n"),
            2,
            "review.status",
        ),
        (GOOD.replace("capstones: []\n", ""), 2, "capstones"),
        (
            GOOD.replace("  methods: [simp]\n", "  methods: 3\n"),
            2,
            "automation.methods",
        ),
        (
            GOOD.replace(
                "main_results:\n  - ProvableContracts.Gelu.gelu_bound\n",
                "main_results: []\n",
            ),
            2,
            "main_results is missing or empty",
        ),
    ];
    for (yaml, sorry, want) in cases {
        let r = judged(&yaml, Some(&summary()), sorry);
        assert!(r.reject, "{want}: passed");
        assert!(
            r.lines.iter().any(|l| l.contains(want)),
            "{want}: {:?}",
            r.lines
        );
    }
}

#[test]
fn a_missing_record_is_red_not_a_default() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("ProvableContracts.lean"),
        "theorem t : True := by sorry\n",
    )
    .unwrap();
    let tree = Tree::load(dir.path()).unwrap();
    let mut r = Report::default();
    judge(dir.path(), &tree, &mut r);
    assert!(r.reject);
    assert!(r.lines.iter().any(|l| l.contains(FILE)), "{:?}", r.lines);
}

#[test]
fn the_sorry_count_ignores_comments() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("ProvableContracts.lean"),
        "-- sorry in a comment\ntheorem t : True := by sorry\n",
    )
    .unwrap();
    assert_eq!(measured_sorry_count(&Tree::load(dir.path()).unwrap()), 1);
}

/// The tracked record, against the tracked summary and tree: the repo's own run of this arm.
#[test]
fn the_repo_record_validates() {
    let lean =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../aprender-contracts-staging/lean");
    let tree = Tree::load(&lean).unwrap();
    let mut r = Report::default();
    judge(&lean, &tree, &mut r);
    assert!(!r.reject, "{:#?}", r.lines);
}
