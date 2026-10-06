use std::collections::BTreeSet;

use super::*;
use crate::lint::strict_test_binding::run_strict_test_binding_gate;

/// A contract whose only rows are the given legacy `falsification:` block, as the corpus writes it.
fn legacy(block: &str) -> Contract {
    let mut c = Contract::default();
    c.falsification = Some(serde_yaml::from_str(block).expect("yaml"));
    c
}

fn index(fns: &[&str]) -> SourceIndex {
    SourceIndex {
        test_fns: fns.iter().map(ToString::to_string).collect(),
        modules: Default::default(),
    }
}

#[allow(clippy::type_complexity)]
fn counts(
    extra: &GateExtra,
) -> (
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    Vec<String>,
    Vec<String>,
) {
    match extra {
        GateExtra::LegacyBinding {
            rows,
            bound,
            shell,
            unbound,
            dangling,
            baselined,
            unbaselined,
            stale,
        } => (
            *rows,
            *bound,
            *shell,
            *unbound,
            *dangling,
            *baselined,
            unbaselined.clone(),
            stale.clone(),
        ),
        other => panic!("expected LegacyBinding, got {other:?}"),
    }
}

/// The lora-target-selection-v1 row as it stands on main: prose in `test:`. It names no test, so it is a hole.
const LORA: &str = r#"
- id: FALSIFY-LORA_TARGET_SELECTION_V1_001
  description: "∀ target ∈ selected: target exists in base model weights"
  test: "Related tests in crates/aprender-train/src/finetune/"
  evidence: "cargo test --lib"
"#;

#[test]
fn each_class_is_read_with_pv_ver_002_rules() {
    let idx = index(&["test_default_config"]);
    let rows = legacy(
        r#"
- {id: B, test: "cargo test -p aprender-train --lib test_default_config"}
- {id: S, test_harness: "grep -q lora book/src/x.md"}
- {id: U, test: "Related tests in crates/aprender-train/src/finetune/"}
- {id: D, test: "cargo test -p aprender-train --lib MUTANT_this_test_fn_does_not_exist_anywhere"}
- {id: N, description: "no binding field at all"}
"#,
    );
    let got: Vec<(String, LegacyClass)> = legacy_rows(&rows)
        .into_iter()
        .map(|(id, ft)| (id, classify_row(&ft, &idx)))
        .collect();
    assert_eq!(
        got,
        vec![
            ("B".into(), LegacyClass::Bound),
            ("S".into(), LegacyClass::Shell),
            ("U".into(), LegacyClass::Unbound),
            (
                "D".into(),
                LegacyClass::Dangling(vec!["MUTANT_this_test_fn_does_not_exist_anywhere".into()])
            ),
            ("N".into(), LegacyClass::Unbound),
        ]
    );
}

#[test]
fn both_legacy_blocks_are_read_and_a_row_without_an_id_is_named_by_its_place() {
    let mut c = legacy("- {test: \"Related tests\"}\n- 7\n");
    c.falsification_conditions =
        Some(serde_yaml::from_str("{id: FC-1, test: \"prose\"}").expect("yaml"));
    let ids: Vec<String> = legacy_rows(&c).into_iter().map(|(id, _)| id).collect();
    assert_eq!(ids, ["falsification[0]", "falsification[1]", "FC-1"]);
}

#[test]
fn a_new_hole_is_reported_and_a_baselined_one_is_not() {
    let contracts = vec![("lora-target-selection-v1".to_string(), legacy(LORA))];
    let key = "lora-target-selection-v1\tFALSIFY-LORA_TARGET_SELECTION_V1_001".to_string();

    let (extra, findings) = measure(&contracts, &index(&[]), &BTreeSet::new());
    assert_eq!(
        counts(&extra),
        (1, 0, 0, 1, 0, 0, vec![key.clone()], vec![])
    );
    assert_eq!(findings.len(), 1, "{findings:?}");
    assert_eq!(findings[0].rule_id, "PV-VER-003");
    assert_eq!(findings[0].severity, RuleSeverity::Info);
    assert_eq!(findings[0].file, "contracts/lora-target-selection-v1.yaml");
    assert!(
        findings[0]
            .message
            .contains("FALSIFY-LORA_TARGET_SELECTION_V1_001"),
        "{}",
        findings[0].message
    );

    let (extra, findings) = measure(&contracts, &index(&[]), &BTreeSet::from([key]));
    assert_eq!(counts(&extra), (1, 0, 0, 1, 0, 1, vec![], vec![]));
    assert!(findings.is_empty(), "{findings:?}");
}

#[test]
fn a_baseline_line_that_is_no_longer_a_hole_is_stale() {
    // wmainred's la-72 fix: row 002 bound to a real test. Its baseline line must then read stale, so the ratchet
    // only tightens.
    let contracts = vec![(
        "lora-target-selection-v1".to_string(),
        legacy("- {id: R2, test: \"cargo test -p aprender-train --lib test_default_config\"}\n"),
    )];
    let line = "lora-target-selection-v1\tR2".to_string();
    let (extra, findings) = measure(
        &contracts,
        &index(&["test_default_config"]),
        &BTreeSet::from([line.clone()]),
    );
    assert_eq!(counts(&extra), (1, 1, 0, 0, 0, 0, vec![], vec![line]));
    assert!(findings.is_empty());
}

#[test]
fn the_baseline_file_skips_comments_and_blank_lines() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::create_dir_all(dir.path().join("scripts")).expect("mkdir");
    std::fs::write(
        dir.path().join(BASELINE_REL_PATH),
        "# header\n\na\tB-1\nc\tD-2\n",
    )
    .expect("write");
    assert_eq!(
        read_baseline(dir.path()),
        BTreeSet::from(["a\tB-1".to_string(), "c\tD-2".to_string()])
    );
    assert!(
        read_baseline(&dir.path().join("absent")).is_empty(),
        "a missing baseline hides nothing"
    );
}

/// The claim the ratchet lands on: legacy rows change neither PV-VER-002's counts nor the gate's verdict. A
/// dangling legacy row beside a clean `falsification_tests` row leaves the strict gate passing, at 1 ref / 0 missing.
#[test]
fn legacy_rows_never_move_the_strict_gate_verdict_or_its_counts() {
    let dir = tempfile::tempdir().expect("tempdir");
    let src = dir.path().join("crates").join("apr-cli").join("src");
    std::fs::create_dir_all(&src).expect("mkdir");
    std::fs::write(src.join("t.rs"), "#[test]\nfn test_default_config() {}\n").expect("write");
    let mut c = legacy(
        "- {id: D, test: \"cargo test -p apr-cli --lib MUTANT_this_test_fn_does_not_exist_anywhere\"}\n",
    );
    c.falsification_tests.push(FalsificationTest {
        id: "FALSIFY-OK-001".into(),
        test: Some("cargo test -p apr-cli --lib test_default_config".into()),
        ..FalsificationTest::default()
    });
    let (gate, findings) = run_strict_test_binding_gate(&[("x".to_string(), c)], dir.path(), true);
    assert!(
        gate.passed,
        "strict mode, and the legacy hole does not fail it"
    );
    match gate.detail {
        crate::lint::GateDetail::Verify {
            total_refs,
            existing,
            missing,
        } => {
            assert_eq!((total_refs, existing, missing), (1, 1, 0));
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(
        findings
            .iter()
            .filter(|f| f.rule_id == "PV-VER-002")
            .count(),
        0
    );
    assert_eq!(
        findings
            .iter()
            .filter(|f| f.rule_id == "PV-VER-003")
            .count(),
        1,
        "{findings:?}"
    );
    assert!(
        matches!(
            gate.extra,
            Some(GateExtra::LegacyBinding { dangling: 1, .. })
        ),
        "{:?}",
        gate.extra
    );
}
