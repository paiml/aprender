#![allow(clippy::disallowed_methods)] // unwrap in tests, as the crate's other test files
use super::{judge, parse_args};
use std::path::{Path, PathBuf};

fn dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("rollup-shape-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn put(d: &Path, name: &str, bytes: &[u8]) -> PathBuf {
    let p = d.join(name);
    std::fs::write(&p, bytes).unwrap();
    p
}

fn why(d: &Path, name: &str, text: &str) -> String {
    judge(&put(d, name, text.as_bytes()), None)
        .unwrap()
        .unwrap_or_default()
}

/// (file, text, substring of the refusal). Every row of the Python selftest carries over.
const REFUSED: &[(&str, &str, &str)] = &[
    (
        "PMAT-530.yaml",
        "- id: PMAT-530\n  title: a\n  \"title\": b\n",
        "duplicate key",
    ),
    (
        "PMAT-531.yaml",
        "- id: PMAT-531\n  'title': a\n  title: b\n",
        "duplicate key",
    ),
    (
        "PMAT-502.yaml",
        "- id: PMAT-502\n  title: \"line one\n- id: PMAT-6270\n  line two\"\n",
        "lines opening",
    ),
    (
        "PMAT-503.yaml",
        "- id: PMAT-503\n  title: x\n...\n",
        "document marker",
    ),
    (
        "PMAT-504.yaml",
        "---\n- id: PMAT-504\n  title: x\n",
        "document marker",
    ),
    (
        "PMAT-505.yaml",
        "- id: PMAT-505\n  created: &id001 2026-10-24\n",
        "anchor",
    ),
    (
        "PMAT-506.yaml",
        "- id: PMAT-506\n  created: *id001\n",
        "anchor",
    ),
    (
        "PMAT-507.yaml",
        "\u{feff}---\n- id: PMAT-507\n",
        "byte order mark",
    ),
    (
        "PMAT-508.yaml",
        "- id: 'PMAT-508'  # quoted\n  title: x\n",
        "does not start",
    ),
    ("PMAT-509.yml", "- id: PMAT-509\n", "<id>.yaml"),
    (
        "PMAT-510.yaml",
        "- id: PMAT-510\n  ? [a]\n  : 1\n",
        "unparseable",
    ),
    (
        "PMAT-513.yaml",
        "- id: PMAT-513\r  title: x\n",
        "carriage return",
    ),
    (
        "PMAT-516.yaml",
        "- id: PMAT-516\n  title: a\n  title: b\n",
        "duplicate key",
    ),
    (
        "PMAT-517.yaml",
        "- id: PMAT-517\n  title: a\n- title: second\n",
        "2 top-level items",
    ),
    (
        "PMAT-518.yaml",
        "- id: PMAT-999\n  title: a\n",
        "does not start",
    ),
    (
        "PMAT-519.yaml",
        "- id: PMAT-519\n  title: [a, {b: 1, b: 2}]\n",
        "duplicate key",
    ),
    (
        "PMAT-520.yaml",
        "- id: PMAT-520\n  tags: [a, b]\n- x\n",
        "2 top-level items",
    ),
    ("PMAT-521.yaml", "just a scalar\n", "lines opening"),
];

#[test]
fn refused_shapes_name_their_reason() {
    let d = dir("refused");
    for (name, text, want) in REFUSED {
        let got = why(&d, name, text);
        assert!(got.contains(want), "{name}: want '{want}' in '{got}'");
    }
}

#[test]
fn unicode_line_separators_are_refused() {
    let d = dir("sep");
    for c in [
        "\u{2028}", "\u{2029}", "\u{85}", "\x0b", "\x0c", "\x1c", "\x1d", "\x1e",
    ] {
        let got = why(
            &d,
            "PMAT-515.yaml",
            &format!("- id: PMAT-515\n  title: a{c}- id: PMAT-6270\n"),
        );
        assert!(got.contains("line separator"), "{c:?}: {got}");
    }
}

#[test]
fn plain_fragments_pass() {
    let d = dir("ok");
    assert_eq!(why(&d, "PMAT-501.yaml", "- id: PMAT-501\n  title: b\n"), "");
    // A last line without a newline, a nested sequence and a repeated key in a DIFFERENT map are fine.
    assert_eq!(
        why(
            &d,
            "PMAT-500.yaml",
            "- id: PMAT-500\n  a: {x: 1}\n  b: {x: 2}\n  c: [1, 2]\n  # tail"
        ),
        ""
    );
}

#[test]
fn bad_utf8_is_a_refusal_not_a_crash() {
    let d = dir("utf8");
    let p = put(&d, "PMAT-514.yaml", b"- id: PMAT-514\n  title: \xff\xfe\n");
    assert!(judge(&p, None).unwrap().unwrap().contains("UTF-8"));
}

#[test]
fn unreadable_and_missing_base_are_not_measured() {
    let d = dir("rc2");
    assert!(judge(&d.join("PMAT-511.yaml"), None).is_err());
    std::fs::create_dir_all(d.join("PMAT-512.yaml")).unwrap();
    assert!(judge(&d.join("PMAT-512.yaml"), None).is_err());
    let ok = put(&d, "PMAT-501.yaml", b"- id: PMAT-501\n");
    assert!(judge(&ok, Some(&d.join("nope").join("roadmap.yaml"))).is_err());
}

#[test]
fn a_fragment_that_dangles_an_alias_is_refused() {
    let d = dir("agg");
    let rm = put(
        &d,
        "roadmap.yaml",
        b"roadmap:\n- id: PMAT-600\n  created: &id001 x\n- id: PMAT-601\n  updated: *id001\n",
    );
    let bad = put(
        &d,
        "PMAT-600.yaml",
        b"- id: PMAT-600\n  title: replaces the anchor's owner\n",
    );
    let good = put(
        &d,
        "PMAT-602.yaml",
        b"- id: PMAT-602\n  title: leaves every alias alone\n",
    );
    let got = judge(&bad, Some(&rm)).unwrap().unwrap_or_default();
    assert!(got.contains("aggregate unparseable"), "{got}");
    assert_eq!(judge(&good, Some(&rm)).unwrap(), None);
}

/// Acceptance criterion 4 of PMAT-6300: every existing entry passes except the five a strict YAML 1.2
/// parser refuses though PyYAML reads them. They are named here and the set is asserted EXACT, so a
/// sixth refusal fails the test, and so does one of the five becoming valid (delete it from the list).
const STRICT_YAML_REFUSED: [&str; 5] = [
    "PMAT-3699",
    "PMAT-3731",
    "PMAT-3732",
    "PMAT-3740",
    "PMAT-3741",
];

#[test]
fn every_existing_entry_passes() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/roadmaps");
    let (mut n, mut refused) = (0, Vec::new());
    for e in std::fs::read_dir(root.join("entries")).expect("entries dir") {
        let p = e.expect("dir entry").path();
        if judge(&p, Some(&root.join("roadmap.yaml")))
            .expect("measured")
            .is_some()
        {
            refused.push(p.file_stem().expect("stem").to_string_lossy().into_owned());
        }
        n += 1;
    }
    refused.sort();
    assert!(n > 100, "only {n} entries seen");
    assert_eq!(refused, STRICT_YAML_REFUSED, "the refused set changed");
}

#[test]
fn arguments_are_strict() {
    let a = |s: &[&str]| s.iter().map(|x| (*x).to_string()).collect::<Vec<_>>();
    assert!(parse_args(&a(&["shape", "f.yaml"])).is_ok());
    assert!(parse_args(&a(&["shape", "f.yaml", "--roadmap", "r.yaml"])).is_ok());
    for bad in [
        &["shape", "f.yaml", "--roadmap"][..],
        &["shape", "f.yaml", "--roadmap", "r.yaml", "x"],
        &["shape", "f.yaml", "r.yaml"],
        &["shape"],
        &[],
        &["other", "f.yaml"],
    ] {
        assert!(parse_args(&a(bad)).is_err(), "{bad:?}");
    }
}
