//! The old `--selftest` rows as unit tests (the order rows that shell out to the repo checker
//! live in `merge_selftest`; here the pure driver is judged directly).

use super::*;

const H: &str = "roadmap_version: '1.0'\nroadmap:\n";
const E1: &str = "- id: PMAT-1\n  a: 1\n";
const E2: &str = "- id: PMAT-2\n  a: 2\n";
const E3: &str = "- id: PMAT-3\n  a: 3\n";

fn ids(t: &str) -> Vec<String> {
    split_entries(t).1.into_iter().map(|e| e.0).collect()
}

#[test]
fn both_append_keeps_both_ascending() {
    let m = merge(
        &format!("{H}{E1}"),
        &format!("{H}{E1}{E3}"),
        &format!("{H}{E1}{E2}"),
        false,
    )
    .unwrap();
    assert_eq!(ids(&m), ["PMAT-1", "PMAT-2", "PMAT-3"]);
}

#[test]
fn one_sided_edit_is_taken_verbatim() {
    let edited = "- id: PMAT-1\n  a: 11\n  note: x\n";
    let m = merge(
        &format!("{H}{E1}"),
        &format!("{H}{edited}"),
        &format!("{H}{E1}"),
        false,
    )
    .unwrap();
    assert!(m.contains(edited));
    let m = merge(
        &format!("{H}{E1}"),
        &format!("{H}{E1}"),
        &format!("{H}{edited}"),
        false,
    )
    .unwrap();
    assert!(m.contains(edited));
}

#[test]
fn divergent_edit_conflicts_naming_the_id() {
    let a = "- id: PMAT-1\n  a: 11\n";
    let b = "- id: PMAT-1\n  a: 12\n";
    let e = merge(
        &format!("{H}{E1}"),
        &format!("{H}{a}"),
        &format!("{H}{b}"),
        false,
    )
    .unwrap_err();
    assert!(e.0.contains("PMAT-1"));
}

#[test]
fn identical_edit_on_both_sides_is_not_a_conflict() {
    let a = "- id: PMAT-1\n  a: 11\n";
    let m = merge(
        &format!("{H}{E1}"),
        &format!("{H}{a}"),
        &format!("{H}{a}"),
        false,
    )
    .unwrap();
    assert!(m.contains(a));
}

#[test]
fn uncontested_deletion_removes_the_entry() {
    let m = merge(
        &format!("{H}{E1}{E2}"),
        &format!("{H}{E1}"),
        &format!("{H}{E1}{E2}"),
        false,
    )
    .unwrap();
    assert_eq!(ids(&m), ["PMAT-1"]);
}

#[test]
fn deleted_versus_edited_conflicts() {
    let e2 = "- id: PMAT-2\n  a: 22\n";
    let e = merge(
        &format!("{H}{E1}{E2}"),
        &format!("{H}{E1}"),
        &format!("{H}{E1}{e2}"),
        false,
    )
    .unwrap_err();
    assert!(e.0.contains("PMAT-2"));
}

#[test]
fn header_taken_from_the_changed_side_and_conflicts_when_both_change() {
    let h2 = "roadmap_version: '2.0'\nroadmap:\n";
    let m = merge(
        &format!("{H}{E1}"),
        &format!("{H}{E1}"),
        &format!("{h2}{E1}"),
        false,
    )
    .unwrap();
    assert!(m.starts_with(h2));
    let h3 = "roadmap_version: '3.0'\nroadmap:\n";
    assert!(merge(
        &format!("{H}{E1}"),
        &format!("{h3}{E1}"),
        &format!("{h2}{E1}"),
        false
    )
    .is_err());
}

#[test]
fn unsorted_tail_appends_are_reseated() {
    let (e4, e5) = ("- id: PMAT-4\n  a: 4\n", "- id: PMAT-5\n  a: 5\n");
    let m = merge(
        &format!("{H}{E1}{e5}"),
        &format!("{H}{E1}{e5}{E3}"),
        &format!("{H}{E1}{e5}{e4}"),
        false,
    )
    .unwrap();
    assert_eq!(ids(&m), ["PMAT-1", "PMAT-3", "PMAT-4", "PMAT-5"]);
}

#[test]
fn mutation_hook_sorts_numerals_as_strings() {
    let e10 = "- id: PMAT-10\n  a: 10\n";
    let (b, o, t) = (
        format!("{H}{E1}"),
        format!("{H}{E1}{e10}"),
        format!("{H}{E1}{E2}"),
    );
    assert_eq!(
        ids(&merge(&b, &o, &t, false).unwrap()),
        ["PMAT-1", "PMAT-2", "PMAT-10"]
    );
    assert_eq!(
        ids(&merge(&b, &o, &t, true).unwrap()),
        ["PMAT-1", "PMAT-10", "PMAT-2"]
    );
}

#[test]
fn legacy_ids_keep_their_slot() {
    let leg = "- id: some legacy thing\n  a: 0\n";
    let m = merge(
        &format!("{H}{E3}{leg}"),
        &format!("{H}{E3}{leg}{E1}"),
        &format!("{H}{E3}{leg}"),
        false,
    )
    .unwrap();
    assert_eq!(
        ids(&m),
        ["PMAT-1", "some legacy thing", "PMAT-3"].map(String::from)
    );
    assert!(
        ids(&m).iter().position(|i| i == "PMAT-1") < ids(&m).iter().position(|i| i == "PMAT-3")
    );
}
