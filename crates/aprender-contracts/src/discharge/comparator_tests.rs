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
