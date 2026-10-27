//! Fixture-only falsifiers for the README render (#6394): tempdirs, no network, no live repo state.

use super::*;

fn contracts() -> tempfile::TempDir {
    let d = tempfile::tempdir().expect("tempdir");
    std::fs::create_dir_all(d.path().join("legacy")).expect("mkdir");
    for (rel, body) in [
        ("a.yaml", "name: a\n"),
        ("b.yaml", "name: b\n"),
        ("legacy/old.yaml", "name: old\n"),
        (".hidden.yaml", "name: h\n"),
        ("ontology.yaml", "name: sigma\n"),
    ] {
        std::fs::write(d.path().join(rel), body).expect("w");
    }
    d
}

fn readme(count: u64, fm: u64, extra: &str) -> String {
    format!(
        "---\ncontract_count: {fm}\n---\n# T\n\nWe carry {START}{count}{END} contracts.\n{extra}"
    )
}

#[test]
fn walked_count_applies_the_readme_sync_exclusions() {
    // legacy/, dotfiles and ontology.yaml do not count.
    assert_eq!(walked_count(contracts().path()), Ok(2));
}

#[test]
fn walked_count_of_an_empty_corpus_is_a_refusal_not_zero() {
    let d = tempfile::tempdir().expect("tempdir");
    let e = walked_count(d.path()).expect_err("0 files is a broken measurement");
    assert_eq!(e.code, 2);
}

#[test]
fn render_touches_only_the_blocks_and_the_frontmatter_line() {
    let t = readme(9, 9, "contract_count: 5 in prose is untouched\n");
    let r = render(&t, 2, 3);
    assert!(r.contains(&format!("{START}2{END}")));
    assert!(r.contains("\ncontract_count: 3\n"));
    assert!(r.contains("contract_count: 5 in prose is untouched"));
    assert_eq!(render(&r, 2, 3), r, "idempotent");
}

#[test]
fn render_replaces_a_hand_edited_block_body_and_skips_an_unclosed_marker() {
    let edited = format!("{START}many{END}");
    assert_eq!(render(&edited, 4, 4), format!("{START}4{END}"));
    let unclosed = format!("{START}7 and no end");
    assert_eq!(render(&unclosed, 4, 4), unclosed);
    assert_eq!(block_count(&format!("{edited}{unclosed}")), 2);
}
