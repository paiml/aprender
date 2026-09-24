use super::*;

fn n(src: &str) -> u64 {
    count_source("t.rs", src).unwrap()
}

fn count_of(files: &[(&str, u64)]) -> Count {
    let files: BTreeMap<String, u64> = files.iter().map(|(p, c)| ((*p).to_owned(), *c)).collect();
    Count { total: files.values().sum(), files }
}

#[test]
fn counts_calls_at_every_depth() {
    let src = r"
        #[kani::proof]
        fn h() {
            let n: usize = kani::any();
            kani::assume(n >= 1);
            if n > 2 { ::kani::assume(n < 9); }
            kani :: assume(n != 5);
        }
        macro_rules! bound { ($x:expr) => { kani::assume($x) }; }
    ";
    assert_eq!(n(src), 4);
}

#[test]
fn comments_and_strings_never_count() {
    let src = r#"
        // kani::assume(x);
        /* kani::assume(y); */
        /// kani::assume(z) in a doc comment
        fn f() -> &'static str { "kani::assume(w)" }
        const R: &str = r"kani::assume(v)";
    "#;
    assert_eq!(n(src), 0);
}

#[test]
fn a_different_path_does_not_count() {
    assert_eq!(n("fn f() { kani::any(); other::assume(x); assume(y); kani::assume_unchecked(z); }"), 0);
}

#[test]
fn a_group_boundary_breaks_the_sequence() {
    // `kani` closes one group and `::assume` opens outside it: not a path.
    assert_eq!(n("fn f() { (kani)::assume(x); }"), 0);
}

#[test]
fn a_use_import_of_assume_is_refused_in_every_shape() {
    for src in [
        "use kani::assume;",
        "use kani::{any, assume};",
        "use ::kani::assume as a;",
        "mod m { use kani::{self, assume}; }",
    ] {
        match count_source("t.rs", src) {
            Err(CountError::AliasedImport { path }) => assert_eq!(path, "t.rs"),
            other => panic!("{src:?}: {other:?}"),
        }
    }
}

#[test]
fn a_use_of_kani_without_assume_is_accepted() {
    assert_eq!(n("use kani::any; fn f() { let _ = any::<u8>(); kani::assume(true); }"), 1);
}

#[test]
fn a_file_that_does_not_tokenize_is_an_error_naming_it() {
    match count_source("bad.rs", "fn f() { \"unterminated }") {
        Err(CountError::Tokenize { path, .. }) => assert_eq!(path, "bad.rs"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_rise_is_rejected_per_file_even_when_the_total_is_unchanged() {
    let base = baseline_of(&count_of(&[("a.rs", 5)]), "make kani-ratchet");
    // Five assumes moved from a.rs to b.rs: the total holds, b.rs rose from 0.
    let moved = count_of(&[("b.rs", 5)]);
    assert_eq!(rises(&moved, &base), vec![Rise { path: "b.rs".into(), was: 0, now: 5 }]);
}

#[test]
fn a_fall_or_a_removed_file_is_not_a_rise() {
    let base = baseline_of(&count_of(&[("a.rs", 5), ("b.rs", 2)]), "make kani-ratchet");
    assert!(rises(&count_of(&[("a.rs", 4)]), &base).is_empty());
    assert!(rises(&count_of(&[("a.rs", 5), ("b.rs", 2)]), &base).is_empty());
}

#[test]
fn one_more_assume_in_a_listed_file_is_a_rise() {
    let base = baseline_of(&count_of(&[("a.rs", 5)]), "make kani-ratchet");
    assert_eq!(rises(&count_of(&[("a.rs", 6)]), &base), vec![Rise { path: "a.rs".into(), was: 5, now: 6 }]);
}

#[test]
fn an_inconsistent_baseline_is_refused() {
    let mut b = baseline_of(&count_of(&[("a.rs", 5)]), "make kani-ratchet");
    assert!(b.consistent().is_ok());
    b.total = 4;
    assert!(b.consistent().unwrap_err().contains("not the sum"), "{b:?}");
    b.total = 5;
    b.files.insert("z.rs".into(), 0);
    assert!(b.consistent().unwrap_err().contains("at 0"), "{b:?}");
}

#[test]
fn baseline_json_has_exactly_the_three_fields_and_refuses_others() {
    let b = baseline_of(&count_of(&[("a.rs", 2)]), "make kani-ratchet");
    let v: serde_json::Value = serde_json::to_value(&b).unwrap();
    // Sorted: the workspace build unifies serde_json's `preserve_order`, so
    // the map's order is the struct's there and alphabetical under `-p`.
    let mut keys: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(keys, ["command", "files", "total"]);
    let extra = r#"{"command":"c","total":0,"files":{},"extra":1}"#;
    assert!(serde_json::from_str::<Baseline>(extra).is_err());
}

#[test]
fn count_tree_keys_relative_paths_and_skips_target_and_hidden() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    for (rel, body) in [
        ("a/src/lib.rs", "fn f() { kani::assume(true); kani::assume(false); }"),
        ("a/src/none.rs", "fn g() {}"),
        ("a/target/debug/gen.rs", "fn h() { kani::assume(true); }"),
        (".hidden/x.rs", "fn i() { kani::assume(true); }"),
        ("b/notes.txt", "kani::assume(true)"),
    ] {
        let p = root.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, body).unwrap();
    }
    let c = count_tree(root).unwrap();
    assert_eq!(c, count_of(&[("a/src/lib.rs", 2)]));
}

#[test]
fn count_tree_fails_closed_on_an_untokenizable_file() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("ok.rs"), "fn f() { kani::assume(true); }").unwrap();
    std::fs::write(dir.path().join("bad.rs"), "fn f() { \"open").unwrap();
    assert!(matches!(count_tree(dir.path()), Err(CountError::Tokenize { path, .. }) if path == "bad.rs"));
}
