use super::*;

/// A 64-hex sha256-shaped hash, distinct per `n`.
fn h(n: u8) -> String {
    format!("{n:02x}").repeat(32)
}

fn row(name: &str, ch: Option<String>, sol: Option<String>, axioms: Option<&[&str]>) -> Row {
    Row {
        name: name.to_string(),
        challenge_type_hash: ch,
        solution_type_hash: sol,
        defeq_instances: None,
        axioms: axioms.map(|a| a.iter().map(|s| (*s).to_string()).collect()),
    }
}

fn closed(name: &str) -> Row {
    row(name, Some(h(1)), Some(h(1)), Some(&["propext"]))
}

fn judge(rows: &[Row]) -> (Report, Closure) {
    let mut r = Report::default();
    let c = judge_rows(rows, &mut r);
    (r, c)
}

fn fails(r: &Report) -> Vec<&str> {
    r.lines
        .iter()
        .filter(|l| l.starts_with("FAIL"))
        .map(String::as_str)
        .collect()
}

#[test]
fn challenge_decl_and_solution_of_round_trip() {
    for f in ["ProvableContracts.Gelu.gelu_bound", "a", "A.B.c'"] {
        let ch = challenge_decl(f);
        assert_eq!(ch, format!("PvlChallenge.{f}"));
        assert_eq!(solution_of(&ch), Some(f));
    }
}

#[test]
fn solution_of_rejects_what_is_not_a_challenge() {
    for s in [
        "PvlChallenge",
        "PvlChallenge.",
        "PvlChallengeX.f",
        "ProvableContracts.f",
        "",
    ] {
        assert_eq!(solution_of(s), None, "{s:?}");
    }
}

#[test]
fn parse_rows_reads_the_comparator_shape_and_skips_blank_lines() {
    let c = h(1);
    let out = format!(
        "{{\"name\": \"A.f\", \"challenge_type_hash\": \"{c}\", \"solution_type_hash\": \"{c}\", \"axioms\": [\"propext\"]}}\n\n\
         {{\"name\": \"A.g\", \"challenge_type_hash\": \"{c}\", \"solution_type_hash\": null, \"axioms\": null}}\n"
    );
    let rows = parse_rows(&out).expect("two rows");
    assert_eq!(
        rows,
        vec![
            row("A.f", Some(c.clone()), Some(c.clone()), Some(&["propext"])),
            row("A.g", Some(c), None, None),
        ]
    );
}

#[test]
fn parse_rows_names_the_bad_line_and_judges_nothing() {
    let err = parse_rows("\n{\"name\": \"A.f\"}\nnot json\n").expect_err("line 3 is not a row");
    assert!(err.contains("line 3"), "{err}");
    assert!(err.contains("not json"), "{err}");
    let err = parse_rows("{\"axioms\": []}").expect_err("a row without a name");
    assert!(err.contains("line 1"), "{err}");
}

#[test]
fn a_matching_sorry_free_row_is_closed() {
    let (r, c) = judge(&[closed("A.f"), closed("A.g")]);
    assert!(!r.reject && r.decline.is_none(), "{:?}", r.lines);
    assert_eq!(
        c,
        Closure {
            closed: 2,
            total: 2
        }
    );
    assert!(r
        .lines
        .contains(&"COMPARATOR 2/2 challenge(s) closed".to_string()));
}

#[test]
fn a_different_statement_is_a_mismatch_naming_both_hashes() {
    let (r, c) = judge(&[row("A.f", Some(h(1)), Some(h(2)), Some(&[]))]);
    assert!(r.reject);
    assert_eq!(
        c,
        Closure {
            closed: 0,
            total: 1
        }
    );
    let f = fails(&r);
    assert_eq!(f.len(), 1, "{f:?}");
    assert!(f[0].starts_with("FAIL  MISMATCH A.f"), "{}", f[0]);
    assert!(f[0].contains(&h(1)) && f[0].contains(&h(2)), "{}", f[0]);
    assert!(f[0].contains("PvlChallenge.A.f"), "{}", f[0]);
}

#[test]
fn a_challenge_with_no_solution_is_missing_root() {
    let (r, c) = judge(&[row("A.f", Some(h(1)), None, None), closed("A.g")]);
    assert!(r.reject);
    assert_eq!(
        c,
        Closure {
            closed: 1,
            total: 2
        }
    );
    let f = fails(&r);
    assert_eq!(f.len(), 1, "{f:?}");
    assert!(f[0].starts_with("FAIL  MISSING-ROOT A.f"), "{}", f[0]);
}

#[test]
fn a_matching_solution_on_sorry_closes_nothing() {
    let (r, c) = judge(&[row(
        "A.f",
        Some(h(1)),
        Some(h(1)),
        Some(&["propext", SORRY_AXIOM]),
    )]);
    assert!(r.reject);
    assert_eq!(c.closed, 0);
    let f = fails(&r);
    assert_eq!(f.len(), 1, "{f:?}");
    assert!(f[0].starts_with("FAIL  SORRY A.f"), "{}", f[0]);
}

#[test]
fn an_axiom_that_merely_contains_sorry_is_not_sorry() {
    let (r, c) = judge(&[row("A.f", Some(h(1)), Some(h(1)), Some(&["sorryAxLike"]))]);
    assert!(!r.reject, "{:?}", r.lines);
    assert_eq!(c.closed, 1);
}

#[test]
fn a_row_reported_twice_is_a_duplicate_counted_once_as_closed() {
    let (r, c) = judge(&[closed("A.f"), closed("A.f")]);
    assert!(r.reject);
    assert_eq!(
        c,
        Closure {
            closed: 1,
            total: 2
        }
    );
    let f = fails(&r);
    assert_eq!(f.len(), 1, "{f:?}");
    assert!(
        f[0].starts_with("FAIL  DUPLICATE PvlChallenge.A.f"),
        "{}",
        f[0]
    );
}

#[test]
fn a_row_without_a_challenge_hash_is_malformed() {
    let (r, c) = judge(&[row("A.f", None, Some(h(1)), Some(&[]))]);
    assert!(r.reject);
    assert_eq!(c.closed, 0);
    assert!(fails(&r)[0].starts_with("FAIL  MALFORMED PvlChallenge.A.f"));
}

#[test]
fn a_solution_without_an_axioms_list_is_malformed_not_closed() {
    let (r, c) = judge(&[row("A.f", Some(h(1)), Some(h(1)), None)]);
    assert!(r.reject);
    assert_eq!(c.closed, 0);
    let f = fails(&r);
    assert_eq!(f.len(), 1, "{f:?}");
    assert!(
        f[0].starts_with("FAIL  MALFORMED PvlChallenge.A.f"),
        "{}",
        f[0]
    );
    assert!(f[0].contains("no axioms list"), "{}", f[0]);
}

#[test]
fn zero_rows_declines_and_is_never_a_pass() {
    let (r, c) = judge(&[]);
    assert!(!r.reject);
    assert_eq!(c, Closure::default());
    let d = r.decline.expect("zero rows must decline");
    assert!(d.contains("0 challenge rows"), "{d}");
}

#[test]
fn zero_rows_after_an_earlier_reject_stays_a_reject() {
    let mut r = Report::default();
    r.fail("earlier".to_string());
    let _ = judge_rows(&[], &mut r);
    assert!(r.reject);
    assert!(r.decline.is_none(), "a reject outranks the decline");
}

#[test]
fn challenge_files_lists_only_lean_files_relative_and_sorted() {
    let d = tempfile::tempdir().expect("tempdir");
    assert!(
        challenge_files(d.path()).is_empty(),
        "no Challenge dir is empty"
    );
    let ch = d.path().join(CHALLENGE_DIR);
    std::fs::create_dir_all(ch.join("sub.lean")).expect("mkdir");
    for f in ["z-v1.lean", "a-v1.lean", "notes.md", "b-v1.olean"] {
        std::fs::write(ch.join(f), "").expect("write");
    }
    assert_eq!(
        challenge_files(d.path()),
        vec![
            PathBuf::from("Challenge/a-v1.lean"),
            PathBuf::from("Challenge/z-v1.lean")
        ]
    );
}

fn with_defeq(mut r: Row, d: Option<bool>) -> Row {
    r.defeq_instances = d;
    r
}

/// #4237: `Zero ℤ` through `MulZeroClass` vs `NegZeroClass` — different hashes, defeq at `.instances`: closed,
/// and the line names both hashes so the fingerprint difference stays visible.
#[test]
fn differing_hashes_defeq_at_instances_is_a_match() {
    let (r, c) = judge(&[with_defeq(
        row("A.f", Some(h(1)), Some(h(2)), Some(&["propext"])),
        Some(true),
    )]);
    assert!(!r.reject, "{:?}", r.lines);
    assert_eq!(c.closed, 1);
    let m: Vec<_> = r
        .lines
        .iter()
        .filter(|l| l.starts_with("MATCH(instances) A.f"))
        .collect();
    assert_eq!(m.len(), 1, "{:?}", r.lines);
    assert!(m[0].contains(&h(1)) && m[0].contains(&h(2)), "{}", m[0]);
}

/// The two negative controls of the ruling (an added hypothesis, a changed right-hand side) measure
/// `defeq_instances: false`; an absent measurement is not a pass either.
#[test]
fn differing_hashes_not_defeq_or_unmeasured_is_a_mismatch() {
    for d in [Some(false), None] {
        let (r, c) = judge(&[with_defeq(row("A.f", Some(h(1)), Some(h(2)), Some(&[])), d)]);
        assert!(r.reject, "{d:?}");
        assert_eq!(c.closed, 0, "{d:?}");
        assert!(fails(&r)[0].starts_with("FAIL  MISMATCH A.f"), "{d:?}");
    }
}

/// Defeq never launders a sorry: the SORRY check still runs after a defeq MATCH.
#[test]
fn a_defeq_match_on_sorry_closes_nothing() {
    let (r, c) = judge(&[with_defeq(
        row("A.f", Some(h(1)), Some(h(2)), Some(&[SORRY_AXIOM])),
        Some(true),
    )]);
    assert!(r.reject);
    assert_eq!(c.closed, 0);
    assert!(fails(&r)[0].starts_with("FAIL  SORRY A.f"), "{:?}", r.lines);
}

// #4240: rows are cross-checked against the roots the Challenge files declare.

fn cross(rows: &[Row], roots: &[&str]) -> (Report, Closure) {
    let (mut r, mut c) = judge(rows);
    let roots: BTreeSet<String> = roots.iter().map(|s| (*s).to_string()).collect();
    cross_check(rows, &roots, &mut c, &mut r);
    (r, c)
}

#[test]
fn roots_in_reads_every_challenge_declaration_and_strips_universes() {
    let text = "section\nopen Real\n\
        theorem _root_.PvlChallenge.A.b (x : ℝ) : x = x := sorry\n\
        theorem _root_.PvlChallenge.A.poly.{u v} {α : Sort u} : True := sorry\n\
          theorem _root_.PvlChallenge.A.indented : True := sorry\n\
        theorem A.not_a_challenge : True := sorry\n\
        -- theorem _root_.PvlChallenge.commented : True := sorry\n\
        theorem _root_.Other.C : True := sorry\nend\n";
    let got: Vec<String> = roots_in(text).into_iter().collect();
    assert_eq!(got, vec!["A.b", "A.indented", "A.poly"]);
}

#[test]
fn a_root_with_no_row_is_missing_row_and_counts_in_m() {
    // `rowsOf` dropped `A.internal` (isInternal): n/m must not read 1/1.
    let (r, c) = cross(&[closed("A.b")], &["A.b", "A.internal"]);
    assert!(r.reject, "a dropped row rejects: {:?}", r.lines);
    let f = fails(&r);
    assert_eq!(f.len(), 1, "{f:?}");
    assert!(f[0].contains("MISSING-ROW A.internal"), "{f:?}");
    assert!(f[0].contains("PvlChallenge.A.internal"), "{f:?}");
    assert_eq!(
        c,
        Closure {
            closed: 1,
            total: 2
        }
    );
}

#[test]
fn a_row_with_no_root_is_unexpected_row() {
    let (r, c) = cross(&[closed("A.b"), closed("A.ghost")], &["A.b"]);
    assert!(r.reject);
    let f = fails(&r);
    assert_eq!(f.len(), 1, "{f:?}");
    assert!(f[0].contains("UNEXPECTED-ROW A.ghost"), "{f:?}");
    assert_eq!(c.total, 2);
}

#[test]
fn rows_equal_to_roots_add_no_failure_and_keep_the_count() {
    let (r, c) = cross(&[closed("A.b"), closed("A.c")], &["A.b", "A.c"]);
    assert!(!r.reject, "{:?}", r.lines);
    assert!(fails(&r).is_empty());
    assert_eq!(
        c,
        Closure {
            closed: 2,
            total: 2
        }
    );
    assert!(r
        .lines
        .iter()
        .any(|l| l.contains("2 row(s), 2 declared root(s)")));
}

#[test]
fn zero_rows_against_declared_roots_rejects_not_declines_away() {
    let (r, c) = cross(&[], &["A.b"]);
    assert!(
        r.reject,
        "every declared root missing is a failure, not vacuity"
    );
    assert!(fails(&r)[0].contains("MISSING-ROW A.b"));
    assert_eq!(
        c,
        Closure {
            closed: 0,
            total: 1
        }
    );
}

#[test]
fn a_duplicate_row_is_one_name_in_the_cross_check() {
    let (r, c) = cross(&[closed("A.b"), closed("A.b")], &["A.b"]);
    let f = fails(&r);
    assert_eq!(f.len(), 1, "only the DUPLICATE: {f:?}");
    assert!(f[0].contains("DUPLICATE"));
    assert_eq!(c.total, 1);
}

#[test]
fn expected_roots_reads_the_listed_files_and_an_unreadable_one_is_an_error() {
    let d = tempfile::tempdir().expect("tempdir");
    let ch = d.path().join(CHALLENGE_DIR);
    std::fs::create_dir_all(&ch).expect("mkdir");
    std::fs::write(
        ch.join("a-v1.lean"),
        "theorem _root_.PvlChallenge.X.y : True := sorry\n",
    )
    .expect("w");
    std::fs::write(
        ch.join("b-v1.lean"),
        "theorem _root_.PvlChallenge.Z : True := sorry\n",
    )
    .expect("w");
    let files = challenge_files(d.path());
    let got: Vec<String> = expected_roots(d.path(), &files)
        .expect("roots")
        .into_iter()
        .collect();
    assert_eq!(got, vec!["X.y", "Z"]);
    let gone = vec![PathBuf::from("Challenge/gone-v1.lean")];
    assert!(expected_roots(d.path(), &gone).is_err());
}
