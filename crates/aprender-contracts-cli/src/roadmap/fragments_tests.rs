use super::*;

const BASE: &str = "roadmap_version: '1.0'\nroadmap:\n- id: PMAT-100\n  title: a\n- id: PMAT-300\n  title: c\n- id: LEGACY-THING\n  title: legacy\n";

fn f(id: &str) -> Entry {
    (id.to_string(), format!("- id: {id}\n  title: x\n"))
}

fn ids(t: &str) -> Vec<String> {
    split_entries(t).1.into_iter().map(|e| e.0).collect()
}

#[test]
fn placement_and_idempotence() {
    assert_eq!(aggregate(BASE, &[]).unwrap(), BASE);
    assert_eq!(
        ids(&aggregate(BASE, &[f("PMAT-200")]).unwrap()),
        ["PMAT-100", "PMAT-200", "PMAT-300", "LEGACY-THING"]
    );
    assert_eq!(
        ids(&aggregate(BASE, &[f("PMAT-90")]).unwrap())[0],
        "PMAT-90"
    );
    assert_eq!(
        ids(&aggregate(BASE, &[f("APEX-1")]).unwrap())
            .last()
            .unwrap(),
        "APEX-1"
    );
    let once = aggregate(BASE, &[f("PMAT-200")]).unwrap();
    assert_eq!(aggregate(&once, &[f("PMAT-200")]).unwrap(), once);
}

#[test]
fn duplicate_fragments_are_refused_and_order_is_irrelevant() {
    assert!(aggregate(BASE, &[f("PMAT-9"), f("PMAT-9")])
        .unwrap_err()
        .contains("duplicate id among fragments"));
    let a = aggregate(BASE, &[f("PMAT-150"), f("PMAT-250")]).unwrap();
    assert_eq!(a, aggregate(BASE, &[f("PMAT-250"), f("PMAT-150")]).unwrap());
}

#[test]
fn a_fragment_supersedes_the_base_entry() {
    let out = aggregate(
        BASE,
        &[(
            "PMAT-100".into(),
            "- id: PMAT-100\n  title: superseded\n".into(),
        )],
    )
    .unwrap();
    assert_eq!(ids(&out), ["PMAT-100", "PMAT-300", "LEGACY-THING"]);
    assert!(out.contains("superseded"));
}

#[test]
fn mutated_append_only_placement_is_caught() {
    let out = aggregate_with(BASE, &[f("PMAT-200")], |e, _| e.len()).unwrap();
    assert_ne!(
        ids(&out),
        ["PMAT-100", "PMAT-200", "PMAT-300", "LEGACY-THING"]
    );
}

#[test]
fn census_rejects_prose_ids() {
    let t = "- id: Push completed work to origin/main (5 commits)\n- id: PMAT-1\n";
    let c = census(t);
    assert_eq!(c[&("unsafe", "legacy")].len(), 1);
    assert_eq!(c[&("safe", "prefixN")], ["PMAT-1"]);
}

#[test]
fn anchors_are_inlined_and_definition_dropped() {
    let text = "- id: A-1\n  created: &id001 2026-01-01\n- id: A-2\n  created: *id001\n";
    let anchors = collect_anchors(text);
    assert_eq!(anchors["id001"], "2026-01-01");
    assert_eq!(
        self_contain("  created: &id001 2026-01-01\n", &anchors),
        "  created: 2026-01-01\n"
    );
    assert_eq!(
        self_contain("  created: *id001\n", &anchors),
        "  created: 2026-01-01\n"
    );
}

#[test]
fn changed_reports_added_changed_removed_and_preamble() {
    let head = "roadmap_version: '2'\nroadmap:\n- id: PMAT-100\n  title: z\n- id: PMAT-400\n  title: d\n- id: LEGACY-THING\n  title: legacy\n";
    let (pre, v) = changed_entries(BASE, head);
    assert!(pre);
    assert_eq!(
        v,
        [
            ("CHANGED", "PMAT-100".to_string()),
            ("REMOVED", "PMAT-300".to_string()),
            ("ADDED", "PMAT-400".to_string())
        ]
    );
}

#[test]
fn adopt_moves_an_entry_and_refuses_a_borrowed_anchor() {
    let td = tempfile::tempdir().unwrap();
    let rm = td.path().join("roadmap.yaml");
    let dir = td.path().join("entries");
    std::fs::write(&rm, BASE).unwrap();
    assert_eq!(
        adopt(&["PMAT-300".into()], &rm, &dir).unwrap(),
        ["PMAT-300"]
    );
    assert!(dir.join("PMAT-300.yaml").is_file());
    assert!(adopt(&["NOPE-1".into()], &rm, &dir)
        .unwrap_err()
        .contains("not a top-level entry"));
    let anch = "roadmap:\n- id: A-1\n  created: &id001 2026-01-01\n- id: A-2\n  created: *id001\n";
    std::fs::write(&rm, anch).unwrap();
    assert!(adopt(&["A-1".into()], &rm, &dir)
        .unwrap_err()
        .contains("alias"));
}

#[test]
fn check_verdicts() {
    let out = aggregate(BASE, &[f("PMAT-200")]).unwrap();
    assert_eq!(check(&out, &out, &[f("PMAT-200")]).0, 0);
    assert_eq!(check(BASE, &out, &[f("PMAT-200")]).0, 1);
}
