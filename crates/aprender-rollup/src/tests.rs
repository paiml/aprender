use super::*;

fn cand(n: u64, files: &[&str]) -> Candidate {
    Candidate {
        number: n,
        additions: 10,
        deletions: 5,
        green: true,
        approved: true,
        labels: vec![OPT_IN_LABEL.to_string()],
        files: files.iter().map(|s| (*s).to_string()).collect(),
        head: format!("sha{n}"),
    }
}

#[test]
fn eligible_candidate_passes() {
    assert_eq!(check(&cand(1, &["a.rs"])), None);
}

#[test]
fn each_gate_rejects() {
    let mut c = cand(1, &["a.rs"]);
    c.green = false;
    assert_eq!(check(&c), Some(Reject::NotGreen));
    let mut c = cand(1, &["a.rs"]);
    c.approved = false;
    assert_eq!(check(&c), Some(Reject::NotApproved));
    let mut c = cand(1, &["a.rs"]);
    c.labels.clear();
    assert_eq!(check(&c), Some(Reject::NoOptInLabel));
}

#[test]
fn line_limit_is_additions_plus_deletions_inclusive() {
    let mut c = cand(1, &["a.rs"]);
    c.additions = 150;
    c.deletions = 50;
    assert_eq!(check(&c), None);
    c.deletions = 51;
    assert_eq!(check(&c), Some(Reject::TooLarge(201)));
    c.additions = u32::MAX;
    assert_eq!(check(&c), Some(Reject::TooLarge(u32::MAX)));
}

#[test]
fn forbidden_paths() {
    for p in [".github/workflows/ci.yml", "contracts/x.yaml", "Cargo.lock"] {
        assert_eq!(
            check(&cand(1, &["a.rs", p])),
            Some(Reject::ForbiddenFile(p.to_string()))
        );
    }
    assert!(!is_forbidden("crates/x/Cargo.lock.rs"));
    assert!(!is_forbidden("docs/contracts/x.md"));
}

#[test]
fn select_orders_by_number_and_skips_overlap() {
    let sel = select(&[
        cand(3, &["b.rs"]),
        cand(1, &["a.rs"]),
        cand(2, &["a.rs", "c.rs"]),
    ]);
    assert_eq!(sel.picked, vec![1, 3]);
    assert_eq!(
        sel.rejected,
        vec![(
            2,
            Reject::FileOverlap {
                with: 1,
                file: "a.rs".to_string()
            }
        )]
    );
}

#[test]
fn select_caps_at_three_and_names_the_rest() {
    let cands: Vec<Candidate> = (1..=12).map(|n| cand(n, &[&format!("f{n}.rs")])).collect();
    let sel = select(&cands);
    assert_eq!(sel.picked, (1..=3).collect::<Vec<u64>>());
    assert_eq!(sel.rejected.len(), 9);
    assert!(sel.rejected.iter().all(|(_, r)| *r == Reject::OverCap));
}

#[test]
fn ineligible_never_blocks_later_candidates() {
    let mut bad = cand(1, &["a.rs"]);
    bad.green = false;
    let sel = select(&[bad, cand(2, &["a.rs"])]);
    assert_eq!(sel.picked, vec![2]);
}

#[test]
fn split_halves_and_stops_at_one() {
    assert_eq!(split(&[1, 2, 3, 4]), Some((vec![1, 2], vec![3, 4])));
    assert_eq!(split(&[1, 2, 3]), Some((vec![1], vec![2, 3])));
    assert_eq!(split(&[1]), None);
    assert_eq!(split(&[]), None);
}

#[test]
fn close_only_green_with_sha_on_main() {
    assert_eq!(
        outcome(true, Some("abc")),
        Outcome::Close {
            sha: "abc".to_string()
        }
    );
    assert!(matches!(
        outcome(false, Some("abc")),
        Outcome::KeepOpen { .. }
    ));
    assert!(matches!(outcome(true, None), Outcome::KeepOpen { .. }));
    assert!(matches!(outcome(true, Some("")), Outcome::KeepOpen { .. }));
}

#[test]
fn select_records_the_head_each_pick_was_judged_at() {
    let cands = vec![cand(4, &["a.rs"]), cand(2, &["b.rs"])];
    let sel = select(&cands);
    assert_eq!(sel.picked, [2, 4]);
    assert_eq!(sel.heads, ["sha2", "sha4"]);
}
