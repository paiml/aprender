use super::*;

const NOW: &str = "2026-09-25T12:00:00Z";

fn ev(pr: u64, kind: EventKind, at: &str) -> Event {
    Event {
        repo: "paiml/aprender".into(),
        pr,
        kind,
        at: at.into(),
    }
}

fn ruling(head: &str, verdict: Verdict, ruled_by: &str, at: &str) -> Ruling {
    Ruling {
        repo: "paiml/aprender".into(),
        pr: 7,
        head: head.into(),
        verdict,
        ruled_by: ruled_by.into(),
        at: at.into(),
    }
}

fn one(events: &[Event], now: &str) -> Resolved {
    let r = resolve(events, now).expect("resolves");
    assert_eq!(r.len(), 1, "{r:?}");
    r.into_iter().next().expect("one")
}

/// FALSIFY-TOJ-001: a 3-day-old outcome is not gold. Until the 14 d window
/// has closed the PR reads `pending`, and only day 14 matures it.
#[test]
fn falsify_toj_001_immature_outcome_is_refused_as_gold() {
    let three = one(&[ev(1, EventKind::Merged, "2026-09-22T09:00:00Z")], NOW);
    assert_eq!(three.outcome, Outcome::Pending);
    assert_eq!(three.outcome_matured_at, None);
    assert!(gold(&[three], &[]).expect("gold").is_empty());
    // A 3-day-old revert is known but still inside the window.
    let rev = one(
        &[
            ev(1, EventKind::Merged, "2026-09-21T00:00:00Z"),
            ev(1, EventKind::Reverted, "2026-09-22T00:00:00Z"),
        ],
        NOW,
    );
    assert_eq!(rev.outcome, Outcome::Pending);
    // Days count by UTC date: 09-12 to 09-25 is 13, whatever the hour.
    let d13 = one(
        &[ev(1, EventKind::Merged, "2026-09-12T00:00:00Z")],
        "2026-09-25T23:59:59Z",
    );
    assert_eq!(d13.outcome, Outcome::Pending, "13 days is immature");
    let d14 = one(&[ev(1, EventKind::Merged, "2026-09-11T08:00:00Z")], NOW);
    assert_eq!(d14.outcome, Outcome::Merged);
    assert_eq!(d14.outcome_matured_at.as_deref(), Some("2026-09-25"));
    let g = gold(&[d14], &[]).expect("gold");
    assert_eq!(
        g,
        vec![Label::Outcome {
            group_key: "paiml/aprender#1".into(),
            outcome: Outcome::Merged,
            outcome_matured_at: "2026-09-25".into(),
        }]
    );
}

/// FALSIFY-TOJ-002: the terminal outcome. A revert on day ≤ 14 after the
/// merge is `reverted_le14d`; a later revert or any escape is
/// `regression_escape`; a PR closed without a merge is `closed_unmerged`.
#[test]
fn falsify_toj_002_outcome_follows_the_revert_window() {
    let old = "2026-08-01T00:00:00Z";
    let at = |kind, at| one(&[ev(2, EventKind::Merged, old), ev(2, kind, at)], NOW).outcome;
    assert_eq!(
        at(EventKind::Reverted, "2026-08-15T23:00:00Z"),
        Outcome::RevertedLe14d
    );
    assert_eq!(
        at(EventKind::Reverted, "2026-08-16T00:00:00Z"),
        Outcome::RegressionEscape
    );
    assert_eq!(
        at(EventKind::Escape, "2026-08-03T00:00:00Z"),
        Outcome::RegressionEscape
    );
    assert_eq!(
        one(&[ev(2, EventKind::Closed, old)], NOW).outcome,
        Outcome::ClosedUnmerged
    );
    assert_eq!(
        one(&[ev(2, EventKind::Merged, old)], NOW).outcome,
        Outcome::Merged
    );
    // A merged PR's close event does not make it unmerged.
    assert_eq!(
        one(
            &[ev(2, EventKind::Closed, old), ev(2, EventKind::Merged, old)],
            NOW
        )
        .outcome,
        Outcome::Merged
    );
    // The window runs from the merge, not from an earlier event.
    let r = one(
        &[
            ev(2, EventKind::Reverted, "2026-08-20T00:00:00Z"),
            ev(2, EventKind::Merged, "2026-08-10T00:00:00Z"),
        ],
        NOW,
    );
    assert_eq!(r.outcome, Outcome::RevertedLe14d);
    assert_eq!(r.outcome_matured_at.as_deref(), Some("2026-08-24"));
    // A revert or escape dated before the merge is not of this merge.
    for kind in [EventKind::Reverted, EventKind::Escape] {
        let early = one(
            &[
                ev(2, kind, "2026-08-05T00:00:00Z"),
                ev(2, EventKind::Merged, "2026-08-10T00:00:00Z"),
            ],
            NOW,
        );
        assert_eq!(early.outcome, Outcome::Merged, "{kind:?}");
    }
    assert_eq!(
        serde_json::to_string(&Outcome::RevertedLe14d).expect("ser"),
        "\"reverted_le14d\""
    );
}

/// FALSIFY-TOJ-003: pending is never a negative, and a join that cannot be
/// read is an error, never a guess.
#[test]
fn falsify_toj_003_pending_is_never_a_label_and_bad_joins_refuse() {
    let open = one(&[ev(4, EventKind::Merged, "2026-09-24T00:00:00Z")], NOW);
    assert_eq!(open.outcome, Outcome::Pending);
    assert!(gold(&[open], &[]).expect("gold").is_empty());
    for bad in [
        vec![ev(3, EventKind::Reverted, "2026-08-01T00:00:00Z")],
        vec![ev(3, EventKind::Escape, "2026-08-01T00:00:00Z")],
        vec![
            ev(3, EventKind::Closed, "2026-08-01T00:00:00Z"),
            ev(3, EventKind::Reverted, "2026-08-02T00:00:00Z"),
        ],
        vec![ev(3, EventKind::Merged, "yesterday")],
    ] {
        assert!(resolve(&bad, NOW).is_err(), "{bad:?}");
    }
    assert!(resolve(&[], "now").is_err());
    // A hand-built immature row is refused, not silently dropped.
    let forged = Resolved {
        group_key: "paiml/aprender#9".into(),
        outcome: Outcome::Merged,
        outcome_matured_at: None,
    };
    assert!(gold(&[forged], &[]).is_err());
}

/// FALSIFY-TOJ-004: an HRQ ruling is gold as soon as it is made; the latest
/// ruling on a round wins; a ruling with no ruler is refused.
#[test]
fn falsify_toj_004_hrq_rulings_are_gold() {
    let g = gold(
        &[],
        &[
            ruling("abc", Verdict::Pass, "noah", "2026-09-25T10:00:00Z"),
            ruling("abc", Verdict::Fail, "noah", "2026-09-25T11:00:00Z"),
            ruling("def", Verdict::Pass, "noah", "2026-09-25T09:00:00Z"),
        ],
    )
    .expect("gold");
    assert_eq!(
        g,
        vec![
            Label::HrqRuling {
                group_key: "paiml/aprender#7".into(),
                head: "abc".into(),
                verdict: Verdict::Fail,
                ruled_by: "noah".into(),
                at: "2026-09-25T11:00:00Z".into(),
            },
            Label::HrqRuling {
                group_key: "paiml/aprender#7".into(),
                head: "def".into(),
                verdict: Verdict::Pass,
                ruled_by: "noah".into(),
                at: "2026-09-25T09:00:00Z".into(),
            },
        ]
    );
    let v = serde_json::to_value(&g[0]).expect("ser");
    assert_eq!(v["label_source"], "hrq_ruling");
    assert!(crate::pool::GOLD_SOURCES.contains(&v["label_source"].as_str().expect("str")));
    for bad in [
        ruling("abc", Verdict::Pass, " ", "2026-09-25T10:00:00Z"),
        ruling("abc", Verdict::Pass, "noah", "today"),
    ] {
        assert!(gold(&[], &[bad]).is_err());
    }
}

/// FALSIFY-TOJ-005: the weekly count is the outcomes whose window closed in
/// `[from, to)`, and nothing still pending.
#[test]
fn falsify_toj_005_weekly_matured_count() {
    let rs = resolve(
        &[
            ev(10, EventKind::Merged, "2026-09-04T00:00:00Z"), // matures 09-18: in
            ev(11, EventKind::Closed, "2026-09-10T00:00:00Z"), // matures 09-24: in
            ev(12, EventKind::Merged, "2026-09-11T00:00:00Z"), // matures 09-25: out (to)
            ev(13, EventKind::Merged, "2026-09-03T00:00:00Z"), // matures 09-17: out (from-1)
            ev(14, EventKind::Merged, "2026-09-20T00:00:00Z"), // pending
        ],
        NOW,
    )
    .expect("resolves");
    assert_eq!(rs.len(), 5);
    assert_eq!(matured_between(&rs, "2026-09-18", "2026-09-25"), 2);
    assert_eq!(matured_between(&rs, "2026-09-17", "2026-09-26"), 4);
    let v = serde_json::to_value(gold(&rs, &[]).expect("gold").first().expect("one")).expect("ser");
    assert_eq!(v["label_source"], "outcome");
}

#[test]
fn date_inverts_day() {
    for s in [
        "1970-01-01",
        "2000-02-29",
        "2024-12-31",
        "2026-09-25",
        "2100-03-01",
    ] {
        let d = crate::split_guard::day(s).expect("day");
        assert_eq!(crate::split_guard::date(d), s);
    }
}
