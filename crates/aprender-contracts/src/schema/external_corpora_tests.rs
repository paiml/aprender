use super::*;
use crate::error::Severity;
use std::path::{Path, PathBuf};

/// Repo-root-relative path, resolved from this crate's manifest dir so the
/// tests read the SAME bytes `pv validate` reads.
fn repo_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative)
}

fn read(relative: &str) -> String {
    std::fs::read_to_string(repo_path(relative))
        .unwrap_or_else(|e| panic!("{relative} is readable: {e}"))
}

fn error_rules(violations: &[Violation]) -> Vec<String> {
    violations
        .iter()
        .filter(|v| v.severity == Severity::Error)
        .map(|v| format!("{}: {}", v.rule, v.message))
        .collect()
}

fn rules_of(yaml: &str) -> Vec<String> {
    error_rules(&validate_external_corpora(yaml))
}

fn assert_raises(yaml: &str, rule: &str) {
    let rules = rules_of(yaml);
    assert!(
        rules.iter().any(|r| r.starts_with(rule)),
        "expected {rule}, got: {rules:?}"
    );
}

/// The declaration in the tree validates clean. This is the PMAT-1098 defect:
/// before the kind existed, `pv validate contracts/external-corpora.yaml`
/// answered ``missing field `metadata` `` and took the 0.68.0 T-2 dogfood to
/// NO-GO on its `pv-contracts` row.
#[test]
fn the_tracked_declaration_validates() {
    let rules = rules_of(&read("contracts/external-corpora.yaml"));
    assert!(rules.is_empty(), "{rules:?}");
}

/// ...and the census can read it, which `EXT-CORPORA-009` is the standing
/// promise of. Read through the same parser so the two cannot drift.
#[test]
fn the_tracked_declaration_is_what_the_census_reads() {
    let parsed = parse_external_corpora_str(&read("contracts/external-corpora.yaml"))
        .expect("the census parser reads the tracked declaration");
    assert_eq!(
        parsed.schema.as_deref(),
        Some("ont.paiml.dev/external-corpora/v1alpha1")
    );
    assert_eq!(parsed.corpora.len(), 1);
    let entry = &parsed.corpora[0];
    assert_eq!(entry.name, "provable-contracts");
    assert_eq!(entry.repo.as_deref(), Some("paiml/provable-contracts"));
    assert_eq!(entry.head.as_deref(), Some("626868c240"));
    assert_eq!(entry.n_files, 397);
}

/// The negative fixture: a missing `head` and a typo'd key, both NAMED. The
/// typo is the load-bearing half — serde drops an unknown key, so `n_flies:`
/// would otherwise read as a corpus of 0 and validate clean.
#[test]
fn the_malformed_fixture_is_refused_naming_both_defects() {
    let yaml = read("tests/fixtures/contracts/external-corpora-malformed.yaml");
    let rules = rules_of(&yaml);
    assert!(
        rules
            .iter()
            .any(|r| r.starts_with("EXT-CORPORA-003") && r.contains("head")),
        "the missing `head` must be named: {rules:?}"
    );
    assert!(
        rules
            .iter()
            .any(|r| r.starts_with("EXT-CORPORA-007") && r.contains("n_flies")),
        "the unknown key must be named: {rules:?}"
    );
}

/// An unknown version of the family is REFUSED, not read as v1alpha1 — even
/// though every other line of the fixture is a valid v1alpha1 declaration.
#[test]
fn an_unknown_schema_version_is_refused() {
    let yaml = read("tests/fixtures/contracts/external-corpora-unknown-version.yaml");
    let rules = rules_of(&yaml);
    assert!(
        rules
            .iter()
            .any(|r| r.starts_with("EXT-CORPORA-001") && r.contains("v9")),
        "{rules:?}"
    );
    // The version is the ONLY thing wrong with it — proof the refusal is the
    // version rule firing and not the rest of the document being broken.
    let v1 = yaml.replace(
        "ont.paiml.dev/external-corpora/v9",
        "ont.paiml.dev/external-corpora/v1alpha1",
    );
    assert!(rules_of(&v1).is_empty(), "{:?}", rules_of(&v1));
}

/// The family prefix is exact: a near-miss schema is not this kind.
#[test]
fn schema_family_prefix_case_table() {
    let cases: &[(&str, bool)] = &[
        ("ont.paiml.dev/external-corpora/v1alpha1", true),
        ("ont.paiml.dev/external-corpora/v9", true),
        ("ont.paiml.dev/external-corpora/", true),
        ("ont.paiml.dev/census/v1alpha1", false),
        ("ont.paiml.dev/external-corpus/v1alpha1", false),
        ("external-corpora/v1alpha1", false),
        ("", false),
    ];
    for (schema, expected) in cases {
        assert_eq!(
            is_external_corpora_schema(schema),
            *expected,
            "misjudged {schema:?}"
        );
    }
}

const GOOD: &str = "schema: ont.paiml.dev/external-corpora/v1alpha1\n\
                    corpora:\n\
                    \x20 - name: c\n\
                    \x20   repo: paiml/c\n\
                    \x20   ref: main\n\
                    \x20   head: 626868c240\n\
                    \x20   n_files: 3\n\
                    \x20   counted_by: gh api\n";

/// The base the mutation table mutates is itself clean — without this, every
/// row below could be passing for the wrong reason.
#[test]
fn the_mutation_base_is_clean() {
    assert!(rules_of(GOOD).is_empty(), "{:?}", rules_of(GOOD));
}

/// One mutation per rule, each turning the clean base RED on ITS rule.
#[test]
fn rule_mutation_table() {
    let cases: &[(&str, &str, &str)] = &[
        ("schema: ont.paiml.dev/external-corpora/v1alpha1", "schema: ont.paiml.dev/external-corpora/v2", "EXT-CORPORA-001"),
        ("corpora:", "corpora: []\nunused:", "EXT-CORPORA-002"),
        ("   ref: main\n", "   ref: ''\n", "EXT-CORPORA-003"),
        ("   repo: paiml/c\n", "   repo: paiml\n", "EXT-CORPORA-004"),
        ("   head: 626868c240\n", "   head: main\n", "EXT-CORPORA-005"),
        ("   n_files: 3\n", "   n_files: -1\n", "EXT-CORPORA-006"),
        ("   counted_by: gh api\n", "   counted_by: [a]\n", "EXT-CORPORA-006"),
        ("- name: c\n", "- name: c\n    nam: c\n", "EXT-CORPORA-007"),
    ];
    for (from, to, rule) in cases {
        let mutated = GOOD.replace(from, to);
        assert_ne!(mutated, GOOD, "mutation {from:?} -> {to:?} did not apply");
        assert_raises(&mutated, rule);
    }
}

/// Two entries, one name: the census sorts and reports by name, so a duplicate
/// makes the figure unattributable.
#[test]
fn a_duplicate_corpus_name_is_refused() {
    let doubled = format!(
        "{GOOD}  - name: c\n    repo: paiml/c\n    ref: main\n    head: 626868c240\n    n_files: 4\n    counted_by: gh api\n"
    );
    assert_raises(&doubled, "EXT-CORPORA-008");
}

/// A declaration the RULES pass but the census CANNOT read is still refused —
/// the one binding that keeps `pv validate` and `pv census` from each being
/// green against their own copy of the shape.
#[test]
fn a_declaration_the_census_cannot_read_is_refused() {
    let no_counted_by = GOOD.replace("   counted_by: gh api\n", "");
    assert!(
        parse_external_corpora_str(&no_counted_by).is_err(),
        "the census parser must reject it, else this test proves nothing"
    );
    assert_raises(&no_counted_by, "EXT-CORPORA-009");
}

/// Shapes that are not a declaration at all are refused, never read as empty.
#[test]
fn non_declaration_shapes_are_refused() {
    for yaml in [
        "- a\n- b\n",
        "schema: ont.paiml.dev/external-corpora/v1alpha1\n",
        "corpora:\n  - name: c\n",
        "schema: ont.paiml.dev/external-corpora/v1alpha1\ncorpora:\n  - just-a-string\n",
    ] {
        assert!(!rules_of(yaml).is_empty(), "accepted {yaml:?}");
    }
}
