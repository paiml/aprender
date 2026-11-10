#![allow(clippy::disallowed_methods)] // unwrap in tests, as the crate's other test files
use super::{aggregate, fold, id_value, parse_args, parse_id, read_fragments};
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
