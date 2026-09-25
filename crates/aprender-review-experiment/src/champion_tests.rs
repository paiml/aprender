use super::*;
use crate::receipt::{NotRun, Verdict};

/// 60 test items: 0..30 defects, 30..60 good.
fn rows(verdict: impl Fn(usize) -> Verdict) -> Vec<Scored> {
    (0..60)
        .map(|i| {
            let v = verdict(i);
            Scored {
                id: format!("t{i:02}"),
                defect: i < 30,
                verdict: v,
                localized: false,
                output_sha: v.executed().then(|| format!("o{i}")),
            }
        })
        .collect()
}

/// The right answer on item `i`, or the wrong one.
fn answer(i: usize, right: bool) -> Verdict {
    if (i < 30) == right {
        Verdict::Fail
    } else {
        Verdict::Pass
    }
}

fn pin(w: &str) -> Pin {
    Pin {
        weights_sha256: w.into(),
        prompt_sha256: "p0".into(),
        adapter_sha256: None,
    }
}

/// Champion: flags defects 0..10 only, never a good item (40 correct, precision 1).
fn champ() -> Vec<Scored> {
    rows(|i| answer(i, !(10..30).contains(&i)))
}

/// Challenger: flags every defect, no good item (60 correct; b = 20, c = 0).
fn better() -> Vec<Scored> {
    rows(|i| answer(i, true))
}

fn evidence<'a>(ch: &'a [Scored], cm: &'a [Scored]) -> Evidence<'a> {
    Evidence {
        id: "prompt-v2",
        tier: "B1a",
        pin: Pin {
            prompt_sha256: "p2".into(),
            ..pin("w0")
        },
        test_version: "test-v1",
        challenger: ch,
        champion: cm,
        h1_holds: Some(true),
        h2_holds: Some(true),
        contamination_hits: Some(0),
        p95_s: Some(40.0),
        queue_budget_p95_s: Some(60.0),
    }
}

#[test]
fn falsify_rxc_001_all_five_gates_promote_and_each_one_alone_rejects() {
    let (ch, cm) = (better(), champ());
    let d = decide(&[], &pin("w0"), &evidence(&ch, &cm));
    assert_eq!(d.outcome, Outcome::Promoted, "{:?}", d.reasons);
    assert_eq!((d.mcnemar_b, d.mcnemar_c, d.evaluations_used), (20, 0, 1));

    let same = champ();
    let mut flips: Vec<(&str, Evidence<'_>)> = Vec::new();
    flips.push((
        "1",
        Evidence {
            challenger: &same,
            ..evidence(&ch, &cm)
        },
    ));
    flips.push((
        "3: H1",
        Evidence {
            h1_holds: None,
            ..evidence(&ch, &cm)
        },
    ));
    flips.push((
        "3: H2",
        Evidence {
            h2_holds: Some(false),
            ..evidence(&ch, &cm)
        },
    ));
    flips.push((
        "4:",
        Evidence {
            contamination_hits: Some(1),
            ..evidence(&ch, &cm)
        },
    ));
    flips.push((
        "4:",
        Evidence {
            contamination_hits: None,
            ..evidence(&ch, &cm)
        },
    ));
    flips.push((
        "5:",
        Evidence {
            p95_s: Some(61.0),
            ..evidence(&ch, &cm)
        },
    ));
    flips.push((
        "5:",
        Evidence {
            queue_budget_p95_s: None,
            ..evidence(&ch, &cm)
        },
    ));
    for (gate, e) in flips {
        let d = decide(&[], &pin("w0"), &e);
        assert_eq!(d.outcome, Outcome::Rejected, "gate {gate}");
        assert_eq!(d.reasons.len(), 1, "gate {gate}: {:?}", d.reasons);
        assert!(d.reasons[0].starts_with(gate), "{:?}", d.reasons);
        assert_eq!(
            d.evaluations_used, 1,
            "a rejection still spends an evaluation"
        );
    }
}

#[test]
fn falsify_rxc_001_more_correct_but_less_precise_is_rejected() {
    // Every defect flagged, plus 5 false FAILs: b = 20, c = 5, p ≈ 0.002.
    let (ch, cm) = (rows(|i| answer(i, !(30..35).contains(&i))), champ());
    let d = decide(&[], &pin("w0"), &evidence(&ch, &cm));
    assert!(d.mcnemar_p.is_some_and(|p| p < ALPHA), "{:?}", d.mcnemar_p);
    assert_eq!(d.outcome, Outcome::Rejected);
    assert_eq!(d.reasons.len(), 1);
    assert!(d.reasons[0].starts_with("2:"), "{:?}", d.reasons);
    assert!(d.precision_delta.is_some_and(|x| x < 0.0));
}

#[test]
fn falsify_rxc_002_the_budget_counts_and_refusals_spend_nothing() {
    let (ch, cm) = (better(), champ());
    let mut ledger = Vec::new();
    for n in 0..EVALUATIONS_PER_VERSION {
        let e = Evidence {
            h1_holds: Some(false),
            ..evidence(&ch, &cm)
        };
        let d = decide(&ledger, &pin("w0"), &e);
        assert_eq!(d.evaluations_used, n + 1);
        ledger.push(d);
    }
    let d = decide(&ledger, &pin("w0"), &evidence(&ch, &cm));
    assert_eq!(d.outcome, Outcome::Refused, "the 21st evaluation");
    assert_eq!(d.evaluations_used, EVALUATIONS_PER_VERSION);
    let fresh = decide(
        &ledger,
        &pin("w0"),
        &Evidence {
            test_version: "test-v2",
            ..evidence(&ch, &cm)
        },
    );
    assert_eq!(
        fresh.outcome,
        Outcome::Promoted,
        "a rotated version has its own budget"
    );

    let never = rows(|_| Verdict::NotRun(NotRun::NoDeclaredExecutor));
    let d = decide(&[], &pin("w0"), &evidence(&never, &cm));
    assert_eq!(
        (d.outcome, d.evaluations_used),
        (Outcome::Refused, 0),
        "R-6"
    );
    let d = decide(
        &[],
        &pin("w0"),
        &Evidence {
            pin: pin("w0"),
            ..evidence(&ch, &cm)
        },
    );
    assert_eq!(
        d.outcome,
        Outcome::Refused,
        "the champion cannot challenge itself"
    );
}

#[test]
fn falsify_rxc_002_promotion_moves_the_pin_and_rollback_is_the_previous_sha() {
    let (ch, cm) = (better(), champ());
    let d = decide(&[], &pin("w0"), &evidence(&ch, &cm));
    let ledger = vec![d];
    assert_eq!(champion(&ledger, &pin("w0")).prompt_sha256, "p2");
    assert_eq!(ledger[0].champion_before, pin("w0"), "rollback target");
}

#[test]
fn falsify_rxc_003_two_consecutive_gold_escapes_demote() {
    let esc = |pr: &str, gold: bool, hit: bool| Escape {
        pr: pr.into(),
        gold,
        champion_passed: hit,
        a_voter_failed: hit,
    };
    assert_eq!(
        demotion(&[
            esc("#1", true, true),
            esc("#2", false, false),
            esc("#3", true, true)
        ]),
        Some(("#1".into(), "#3".into())),
        "a silver row neither breaks nor extends the run"
    );
    assert_eq!(
        demotion(&[
            esc("#1", true, true),
            esc("#2", true, false),
            esc("#3", true, true)
        ]),
        None,
        "a gold escape the champion caught breaks the run"
    );
    assert_eq!(demotion(&[esc("#1", true, true)]), None);
}
