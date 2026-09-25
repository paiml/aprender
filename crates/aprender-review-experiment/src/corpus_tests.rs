use super::*;

const MUTANT: &str = "--- crates/x/src/a.rs\n+++ replace f -> bool with true\n@@ -10,7 +10,7 @@\n fn f() -> bool {\n-    x > 1\n+    true /* ~ changed by cargo-mutants ~ */\n }\n";

#[test]
fn sanitize_removes_every_mutation_marker() {
    let s = sanitize_mutant_diff(MUTANT, "crates/x/src/a.rs");
    assert!(names_the_mutation(MUTANT));
    assert!(!names_the_mutation(&s), "{s}");
    assert!(s.starts_with("--- a/crates/x/src/a.rs\n+++ b/crates/x/src/a.rs\n@@ -10,7 +10,7 @@\n"));
    assert!(s.contains("\n+    true\n"));
}

#[test]
fn defect_locs_uses_new_side_lines() {
    let s = sanitize_mutant_diff(MUTANT, "crates/x/src/a.rs");
    // @@ +10: line 10 is context, the '-' is skipped on the new side, '+' is line 11.
    assert_eq!(
        defect_locs(&s),
        vec![Loc {
            file: "crates/x/src/a.rs".into(),
            line: 11
        }]
    );
    let del = "--- a/b.rs\n+++ b/b.rs\n@@ -5,3 +5,2 @@\n a\n-b\n c\n";
    assert_eq!(
        defect_locs(del),
        vec![Loc {
            file: "b.rs".into(),
            line: 6
        }]
    );
}

#[test]
fn review_paths_exclude_tests_benches_examples() {
    assert!(is_review_path("crates/a/src/lib.rs"));
    assert!(is_review_path("scripts/check.sh"));
    for p in [
        "crates/a/tests/x.rs",
        "tests/x.rs",
        "crates/a/src/x_tests.rs",
        "crates/a/src/tests.rs",
        "crates/a/benches/b.rs",
        "crates/a/examples/e.rs",
        "README.md",
        "contracts/a.yaml",
    ] {
        assert!(!is_review_path(p), "{p}");
    }
}

#[test]
fn strata_boundaries() {
    assert_eq!(stratum(1999), Stratum::S);
    assert_eq!(stratum(2000), Stratum::M);
    assert_eq!(stratum(8000), Stratum::M);
    assert_eq!(stratum(8001), Stratum::L);
    let it = Item::new("g1".into(), Class::G, "pr".into(), &"x".repeat(8001));
    assert_eq!((it.approx_tokens, it.stratum), (2001, Stratum::M));
    assert!(it.defect.is_empty());
}

fn items(n: usize, class: Class) -> Vec<Item> {
    (0..n)
        .map(|i| {
            Item::new(
                format!("{class:?}{i:03}"),
                class,
                String::new(),
                &format!("d{i}"),
            )
        })
        .collect()
}

#[test]
fn split_is_stratified_seeded_and_stable() {
    let mut a = items(50, Class::R);
    a.extend(items(10, Class::G));
    let mut b = a.clone();
    b.reverse();
    assign_splits(&mut a, 4354);
    assign_splits(&mut b, 4354);
    assert_eq!(a, b, "input order must not matter");
    let dev = |c: Class| {
        a.iter()
            .filter(|i| i.class == c && i.split == Split::Dev)
            .count()
    };
    assert_eq!((dev(Class::R), dev(Class::G)), (15, 3));
    let mut c = a.clone();
    assign_splits(&mut c, 7);
    assert_ne!(a, c, "a different seed must move the split");
}

#[test]
fn hunks_ignore_line_numbers() {
    let a = "--- a/f\n+++ b/f\n@@ -1,4 +1,4 @@\n x\n-y\n+z\n w\n";
    let b = "--- a/f\n+++ b/f\n@@ -90,4 +91,4 @@ fn g()\n x\n-y\n+z\n w\n";
    assert_eq!(hunk_fingerprints(a), hunk_fingerprints(b));
    assert_eq!(hunk_fingerprints(a).len(), 1);
    assert!(hunk_fingerprints("--- a/f\n+++ b/f\n@@ -1 +1 @@\n-}\n+}\n").is_empty());
}

#[test]
fn manifest_round_trips_and_rejects_garbage() {
    let s = vec![Sealed {
        id: "r1".into(),
        diff_sha256: "a".repeat(64),
        hunks: vec!["b".repeat(64)],
    }];
    let m = render_manifest(&s);
    assert_eq!(parse_manifest(&m), Some(s));
    assert_eq!(parse_manifest("nope\n"), None);
    assert_eq!(
        parse_manifest(&format!("{SCHEME} test-manifest\nr1 short x\n")),
        None
    );
    assert!(corpus_version(&m).starts_with("review-corpus-v1@"));
}

#[test]
fn pick_balanced_round_robins_strata() {
    let mut c = items(10, Class::R);
    c.push(Item::new(
        "m1".into(),
        Class::R,
        String::new(),
        &"x".repeat(9000),
    ));
    c.push(Item::new(
        "l1".into(),
        Class::R,
        String::new(),
        &"x".repeat(40000),
    ));
    let p = pick_balanced(c.clone(), 4, 1);
    assert_eq!(p.len(), 4);
    assert!(p.iter().any(|i| i.stratum == Stratum::M) && p.iter().any(|i| i.stratum == Stratum::L));
    assert_eq!(p, pick_balanced(c.clone(), 4, 1));
    assert_eq!(pick_balanced(c, 99, 1).len(), 12);
}

#[test]
fn comment_only_hunks_are_not_the_fix() {
    let d = "diff --git a/a.rs b/a.rs\nindex 1..2 100644\n--- a/a.rs\n+++ b/a.rs\n@@ -1,3 +1,3 @@\n //! doc\n-//! old path\n+//! new path\n@@ -20,3 +20,3 @@\n fn f() {\n-    x > 1\n+    x >= 1\n }\ndiff --git a/b.rs b/b.rs\n--- a/b.rs\n+++ b/b.rs\n@@ -5,2 +5,2 @@\n-// a\n+// b\n";
    let k = drop_comment_only_hunks(d);
    assert_eq!(
        k,
        "diff --git a/a.rs b/a.rs\nindex 1..2 100644\n--- a/a.rs\n+++ b/a.rs\n@@ -20,3 +20,3 @@\n fn f() {\n-    x > 1\n+    x >= 1\n }\n"
    );
    assert_eq!(
        defect_locs(&k),
        vec![Loc {
            file: "a.rs".into(),
            line: 21
        }]
    );
    assert_eq!(
        drop_comment_only_hunks("--- a/s.sh\n+++ b/s.sh\n@@ -1 +1 @@\n-# x\n+# y\n"),
        ""
    );
}
