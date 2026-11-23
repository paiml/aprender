#![allow(clippy::disallowed_methods)] // unwrap in tests, as the crate's other test files
use super::{
    aggregate, check, fold, id_value, parse_args, parse_cmd, parse_id, read_fragments, Cmd,
};
use std::path::{Path, PathBuf};

const BASE: &str = "# header\nroadmap:\n- id: PMAT-100\n  title: a\n- id: PMAT-300\n  title: c\n- id: LEGACY-THING\n  title: l\n";

fn f(id: &str, t: &str) -> (String, String) {
    (id.into(), format!("- id: {id}\n  title: {t}\n"))
}

fn ids(text: &str) -> Vec<&str> {
    text.lines()
        .filter_map(|l| l.strip_prefix("- id: "))
        .collect()
}

fn dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("rollup-fold-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn no_fragments_is_the_base_byte_for_byte() {
    assert_eq!(aggregate(BASE, &[]).unwrap(), BASE);
    assert_eq!(aggregate("", &[]).unwrap(), "");
}

#[test]
fn placement_is_the_sorted_slot_within_the_prefix() {
    let cases: &[(&str, &[&str])] = &[
        (
            "PMAT-200",
            &["PMAT-100", "PMAT-200", "PMAT-300", "LEGACY-THING"],
        ),
        (
            "PMAT-400",
            &["PMAT-100", "PMAT-300", "LEGACY-THING", "PMAT-400"],
        ),
        (
            "APEX-1",
            &["PMAT-100", "PMAT-300", "LEGACY-THING", "APEX-1"],
        ),
        (
            "FREEFORM",
            &["PMAT-100", "PMAT-300", "LEGACY-THING", "FREEFORM"],
        ),
        (
            "PMAT-90",
            &["PMAT-90", "PMAT-100", "PMAT-300", "LEGACY-THING"],
        ),
        // leading zeros are not magnitude: 0200 sorts as 200
        (
            "PMAT-0200",
            &["PMAT-100", "PMAT-0200", "PMAT-300", "LEGACY-THING"],
        ),
        // wider than any machine integer still compares as a number, so it goes last of its prefix
        (
            "PMAT-99999999999999999999999",
            &[
                "PMAT-100",
                "PMAT-300",
                "LEGACY-THING",
                "PMAT-99999999999999999999999",
            ],
        ),
    ];
    for (id, want) in cases {
        assert_eq!(ids(&aggregate(BASE, &[f(id, "x")]).unwrap()), *want, "{id}");
    }
}

#[test]
fn a_fragment_supersedes_the_base_entry_of_its_id() {
    let out = aggregate(BASE, &[f("PMAT-100", "superseded")]).unwrap();
    assert_eq!(ids(&out), ["PMAT-100", "PMAT-300", "LEGACY-THING"]);
    assert!(out.contains("superseded") && !out.contains("title: a\n"));
}

#[test]
fn duplicate_ids_among_fragments_are_refused() {
    let e = aggregate(BASE, &[f("PMAT-9", "a"), f("PMAT-9", "b")]).unwrap_err();
    assert!(e.contains("duplicate id among fragments: PMAT-9"), "{e}");
}

#[test]
fn folding_is_idempotent_and_order_free() {
    let frag = [f("PMAT-200", "b")];
    let once = aggregate(BASE, &frag).unwrap();
    assert_eq!(aggregate(&once, &frag).unwrap(), once);
    let two = [f("PMAT-150", "x"), f("PMAT-250", "y")];
    let rev = [two[1].clone(), two[0].clone()];
    assert_eq!(
        aggregate(BASE, &two).unwrap(),
        aggregate(BASE, &rev).unwrap()
    );
    assert_eq!(
        ids(&aggregate(BASE, &two).unwrap()),
        [
            "PMAT-100",
            "PMAT-150",
            "PMAT-250",
            "PMAT-300",
            "LEGACY-THING"
        ]
    );
}

#[test]
fn quoted_ids_and_crlf_bytes_survive() {
    let base = "x:\r\n- id: 'it''s'\r\n  t: 1\r\n- id: \"PMAT-5\"\r\n  t: 2\r\n";
    assert_eq!(aggregate(base, &[]).unwrap(), base);
    let out = aggregate(base, &[f("PMAT-5", "new")]).unwrap();
    assert!(out.starts_with("x:\r\n- id: 'it''s'\r\n  t: 1\r\n- id: PMAT-5\n"));
    assert_eq!(out.matches("PMAT-5").count(), 1);
}

#[test]
fn id_shapes() {
    for (id, want) in [
        ("PMAT-5", Some(("PMAT", "5"))),
        ("PMAT-0", Some(("PMAT", ""))),
        ("PMAT-12 (x)", Some(("PMAT", "12"))),
        ("A_b1-007.x", Some(("A_b1", "7"))),
        ("APR-PERF-GATE-001", None),
        ("PMAT-12a", None),
        ("PMAT-12_", None),
        ("PMAT-", None),
        ("1PMAT-5", None),
        ("PMAT", None),
        ("", None),
    ] {
        assert_eq!(parse_id(id), want, "{id:?}");
    }
}

#[test]
fn only_dash_id_lines_at_column_zero_open_an_entry() {
    let base = "p:\n  - id: nested\n- id:\tTAB-1\n  note: |\n    - id: not-an-entry\n- idx: no\n";
    let out = aggregate(base, &[f("TAB-2", "n")]).unwrap();
    assert_eq!(out, format!("{base}- id: TAB-2\n  title: n\n"));
}

#[test]
fn fragments_read_by_filename_and_a_missing_dir_is_empty() {
    let d = dir("read");
    std::fs::write(d.join("PMAT-2.yaml"), "- id: PMAT-2\n").unwrap();
    std::fs::write(d.join("PMAT-10.yaml"), "- id: PMAT-10\n").unwrap();
    std::fs::write(d.join("notes.txt"), "x").unwrap();
    let got: Vec<String> = read_fragments(&d)
        .unwrap()
        .into_iter()
        .map(|(s, _)| s)
        .collect();
    assert_eq!(got, ["PMAT-10", "PMAT-2"]);
    assert!(read_fragments(&d.join("absent")).unwrap().is_empty());
    std::fs::write(d.join("PMAT-3.yaml"), [0xff, 0xfe]).unwrap();
    assert!(read_fragments(&d).unwrap_err().contains("PMAT-3.yaml"));
}

#[test]
fn an_unreadable_base_is_an_error() {
    let d = dir("nobase");
    assert!(fold(&d.join("nope.yaml"), &d)
        .unwrap_err()
        .contains("cannot read"));
}

#[test]
fn args_are_exactly_fold_base_dir() {
    let a = |v: &[&str]| v.iter().map(ToString::to_string).collect::<Vec<_>>();
    assert_eq!(
        parse_args(&a(&["fold", "b", "e"])).unwrap(),
        (PathBuf::from("b"), PathBuf::from("e"))
    );
    for bad in [
        &[][..],
        &["fold"],
        &["fold", "b"],
        &["fold", "b", "e", "x"],
        &["shape", "b", "e"],
    ] {
        assert!(parse_args(&a(bad)).is_err(), "{bad:?}");
    }
}

/// The live tree: the committed aggregate IS fold(frozen base, entries/). Same bytes the Python made.
#[test]
fn the_committed_roadmap_is_the_fold_of_its_base() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let rm = root.join("docs/roadmaps");
    let want = std::fs::read_to_string(rm.join("roadmap.yaml")).unwrap();
    let got = fold(&rm.join("roadmap-base.yaml"), &rm.join("entries")).unwrap();
    assert!(
        got == want,
        "fold(roadmap-base.yaml, entries/) != roadmap.yaml"
    );
}

#[test]
fn id_value_undoes_each_quoting_shape() {
    for (raw, want) in [
        ("'it''s'", "it's"),
        ("\"a\\\"b\"", "a\"b"),
        ("PMAT-5", "PMAT-5"),
        ("'", "'"),
        ("''", ""),
        ("\"\"", ""),
        ("'a\"", "'a\""),
    ] {
        assert_eq!(id_value(raw), want, "{raw:?}");
    }
}

// ---- T1: a fragment is ONE entry whose id is its filename (YAML validity: the Python guard) ----

#[test]
fn a_fragment_with_text_above_its_entry_is_refused() {
    let bad = (
        "PMAT-200".to_string(),
        "title: stray\n- id: PMAT-200\n".to_string(),
    );
    assert!(aggregate(BASE, &[bad])
        .unwrap_err()
        .contains("exactly one entry mapping"));
    let ok = (
        "PMAT-200".to_string(),
        "# a comment\n\n- id: PMAT-200\n".to_string(),
    );
    assert!(aggregate(BASE, &[ok]).is_ok());
}

#[test]
fn a_crlf_or_quoted_fragment_id_still_matches_its_filename() {
    for t in [
        "- id: PMAT-200\r\n  title: a\r\n",
        "- id: 'PMAT-200'\n",
        "- id: \"PMAT-200\"\n",
    ] {
        assert!(
            aggregate(BASE, &[("PMAT-200".into(), t.into())]).is_ok(),
            "{t:?}"
        );
    }
}

#[test]
fn a_fragment_naming_another_id_is_refused() {
    let bad = (
        "PMAT-200".to_string(),
        "- id: PMAT-201\n  title: misfiled\n".to_string(),
    );
    let e = aggregate(BASE, &[bad]).unwrap_err();
    assert!(e.contains("carries id PMAT-201, not its filename"), "{e}");
}

#[test]
fn a_fragment_holding_two_entries_or_a_mapping_is_refused() {
    let two = (
        "PMAT-200".to_string(),
        "- id: PMAT-200\n- id: PMAT-201\n".to_string(),
    );
    let map = ("PMAT-200".to_string(), "id: PMAT-200\n".to_string());
    for bad in [two, map] {
        assert!(aggregate(BASE, &[bad])
            .unwrap_err()
            .contains("must hold exactly one entry mapping"));
    }
}

// ---- T2: `check <roadmap> <entries>` == roadmap_fragments.py `aggregate --check` ----------

fn tree(tag: &str, roadmap: &str, base: Option<&str>, frags: &[(String, String)]) -> PathBuf {
    let d = dir(tag);
    let entries = d.join("entries");
    let _ = std::fs::remove_dir_all(&entries);
    std::fs::create_dir_all(&entries).unwrap();
    std::fs::write(d.join("roadmap.yaml"), roadmap).unwrap();
    let _ = std::fs::remove_file(d.join("roadmap-base.yaml"));
    if let Some(b) = base {
        std::fs::write(d.join("roadmap-base.yaml"), b).unwrap();
    }
    for (id, t) in frags {
        std::fs::write(entries.join(format!("{id}.yaml")), t).unwrap();
    }
    d
}

#[test]
fn check_passes_a_regenerated_roadmap_and_names_the_count() {
    let frag = f("PMAT-200", "b");
    let out = aggregate(BASE, std::slice::from_ref(&frag)).unwrap();
    let d = tree("chk-ok", &out, Some(BASE), &[frag]);
    let msg = check(&d.join("roadmap.yaml"), &d.join("entries")).unwrap();
    assert!(
        msg.contains("== aggregate(1 fragment(s)), idempotent"),
        "{msg}"
    );
}

#[test]
fn check_refuses_a_roadmap_that_was_not_regenerated() {
    let d = tree("chk-drift", BASE, Some(BASE), &[f("PMAT-200", "b")]);
    let e = check(&d.join("roadmap.yaml"), &d.join("entries")).unwrap_err();
    assert!(e.contains("is not what the aggregator produces"), "{e}");
}

#[test]
fn check_without_a_frozen_base_uses_the_roadmap_as_its_own_base() {
    // The scratch-dir shape every guard caller relies on (roadmap_fragments.py base_path).
    let frag = f("PMAT-100", "a");
    let out = aggregate(BASE, std::slice::from_ref(&frag)).unwrap();
    let d = tree("chk-nobase", &out, None, &[frag]);
    assert!(check(&d.join("roadmap.yaml"), &d.join("entries")).is_ok());
}

#[test]
fn check_refuses_a_hand_edited_header_against_the_frozen_base() {
    let tampered = BASE.replace("# header", "# edited");
    let d = tree("chk-header", &tampered, Some(BASE), &[]);
    assert!(check(&d.join("roadmap.yaml"), &d.join("entries")).is_err());
}

#[test]
fn check_and_fold_args_parse_and_nothing_else_does() {
    let a = |v: &[&str]| v.iter().map(ToString::to_string).collect::<Vec<_>>();
    assert_eq!(
        parse_cmd(&a(&["check", "r", "e"])).unwrap(),
        Cmd::Check(PathBuf::from("r"), PathBuf::from("e"))
    );
    assert_eq!(
        parse_cmd(&a(&["fold", "b", "e"])).unwrap(),
        Cmd::Fold(PathBuf::from("b"), PathBuf::from("e"))
    );
    for bad in [&["check", "r"][..], &["chek", "r", "e"], &[]] {
        assert!(parse_cmd(&a(bad)).is_err(), "{bad:?}");
    }
}

// ---- T3: the real tree. Robust to a stale roadmap.yaml: it compares the fold with ITSELF ----

#[test]
fn every_real_fragment_parses_and_the_real_fold_is_idempotent() {
    let docs = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/roadmaps");
    let base_file = if docs.join("roadmap-base.yaml").exists() {
        docs.join("roadmap-base.yaml")
    } else {
        docs.join("roadmap.yaml")
    };
    let frags = read_fragments(&docs.join("entries")).unwrap();
    assert!(
        frags.len() >= 400,
        "only {} fragments: wrong dir?",
        frags.len()
    );
    let out = fold(&base_file, &docs.join("entries")).unwrap();
    assert_eq!(
        aggregate(&out, &frags).unwrap(),
        out,
        "fold(fold(x)) != fold(x)"
    );
    let got = ids(&out);
    for (id, _) in &frags {
        let n = got
            .iter()
            .filter(|g| g.trim_matches(['\'', '"']) == id)
            .count();
        assert_eq!(n, 1, "{id} appears {n} times in the fold");
    }
}
