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

/// `pick_balanced` off-thread with a deadline: a loop that no longer ends when
/// `n` exceeds the population fails here in seconds instead of hanging the run.
fn pick_bounded(c: Vec<Item>, n: usize, seed: u64) -> Vec<Item> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(pick_balanced(c, n, seed));
    });
    rx.recv_timeout(std::time::Duration::from_secs(10))
        .expect("pick_balanced must terminate when n exceeds the population")
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
    assert_eq!(pick_bounded(c, 99, 1).len(), 12);
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

#[test]
fn names_the_mutation_fires_on_either_marker_alone() {
    assert!(names_the_mutation("x /* ~ changed by cargo-mutants ~ */"));
    assert!(names_the_mutation("+++ replace f -> bool with true"));
    assert!(!names_the_mutation("plain diff text"));
}

#[test]
fn defect_locs_counts_context_and_added_lines_on_the_new_side() {
    let d = "--- a/f.rs\n+++ b/f.rs\n@@ -10,4 +10,5 @@\n a\n+b\n+c\n d\n+e\n";
    let lines: Vec<u32> = defect_locs(d).iter().map(|l| l.line).collect();
    assert_eq!(lines, vec![11, 12, 14]);
}

#[test]
fn hash_lines_are_comments_only_in_shell_files() {
    let rs =
        "--- a/x.rs\n+++ b/x.rs\n@@ -1,3 +1,3 @@\n a\n-#[derive(Debug)]\n+#[derive(Clone)]\n b\n";
    assert_eq!(
        drop_comment_only_hunks(rs),
        rs,
        "an attribute is code in .rs"
    );
    let sh = "--- a/x.sh\n+++ b/x.sh\n@@ -1,3 +1,3 @@\n a\n-# old\n+# new\n b\n";
    assert_eq!(
        drop_comment_only_hunks(sh),
        "",
        "a # line is a comment in .sh"
    );
    let sh_code = "--- a/x.sh\n+++ b/x.sh\n@@ -1,3 +1,3 @@\n a\n-echo old\n+echo new\n b\n";
    assert_eq!(drop_comment_only_hunks(sh_code), sh_code);
}

#[test]
fn split_cells_are_selected_by_class_and_stratum_together() {
    // Unequal cell sizes: 10 R and 20 G -> dev = round(0.3 n) = 3 and 6.
    let mut a = items(10, Class::R);
    a.extend(items(20, Class::G));
    a.extend(items(40, Class::P));
    assign_splits(&mut a, 11);
    let dev = |c: Class| {
        a.iter()
            .filter(|i| i.class == c && i.split == Split::Dev)
            .count()
    };
    assert_eq!((dev(Class::R), dev(Class::G), dev(Class::P)), (3, 6, 12));
    // Two strata of one class split independently.
    let mut b = items(10, Class::R);
    for i in 0..10 {
        b.push(Item::new(
            format!("Rm{i:03}"),
            Class::R,
            String::new(),
            &format!("{i}{}", "x".repeat(9000)),
        ));
    }
    assign_splits(&mut b, 11);
    let dev_in = |s: Stratum| {
        b.iter()
            .filter(|i| i.stratum == s && i.split == Split::Dev)
            .count()
    };
    assert_eq!((dev_in(Stratum::S), dev_in(Stratum::M)), (3, 3));
}

#[test]
fn hunks_split_at_every_hunk_and_file_header() {
    let d = "diff --git a/f b/f\n--- a/f\n+++ b/f\n@@ -1,2 +1,2 @@\n a\n-b\n+c\n@@ -9,2 +9,2 @@\n x\n-y\n+z\ndiff --git a/g b/g\n--- a/g\n+++ b/g\n@@ -1,2 +1,2 @@\n p\n-q\n+r\n";
    assert_eq!(
        hunks(d),
        vec![
            " a\n-b\n+c\n".to_string(),
            " x\n-y\n+z\n".to_string(),
            " p\n-q\n+r\n".to_string(),
        ]
    );
}

#[test]
fn fingerprints_need_a_changed_line_and_min_length() {
    let all_context = "@@ -1,4 +1,4 @@\n a\n b\n c\n d\n";
    assert!(hunk_fingerprints(all_context).is_empty(), "no change");
    let changes_only = "@@ -1,4 +1,4 @@\n-a\n-b\n+c\n+d\n";
    assert_eq!(hunk_fingerprints(changes_only).len(), 1);
}

#[test]
fn pick_balanced_never_exceeds_n_and_stops_when_exhausted() {
    let mut c = items(3, Class::R);
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
    for n in 0..=3 {
        assert_eq!(pick_balanced(c.clone(), n, 3).len(), n, "n={n}");
    }
    // Round-robin S, M, L with n=2 stops mid-round: never a third item.
    assert_eq!(pick_balanced(c.clone(), 2, 3).len(), 2);
    // n above the population must terminate (bounded, see `pick_bounded`).
    assert_eq!(pick_bounded(c, 99, 3).len(), 5);
}
