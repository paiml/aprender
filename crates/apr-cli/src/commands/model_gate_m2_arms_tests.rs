//! EXT-30 (aprender#4412): the C7 upstream-stock arm on top of M2.

use super::super::{M2Prereg, ReleaseClass, Suite, SuiteData, Verdict};
use super::*;

const PRE: M2Prereg = M2Prereg::REX_001;
const N: usize = 100;

/// The candidate passes the first 80 of 100 items: Wilson half-width 0.078, powered.
fn cand() -> Vec<bool> {
    (0..N).map(|i| i < 80).collect()
}

/// The candidate paired against `base`.
fn suite(name: &str, base: Option<Vec<bool>>) -> Suite {
    Suite {
        name: name.into(),
        data: SuiteData::Binary {
            candidate: cand(),
            baseline: base,
        },
    }
}

/// A baseline that passes every item the candidate passes, plus `extra` of its failures:
/// `extra` discordant pairs, all in the baseline's favour.
fn above(extra: usize) -> Vec<bool> {
    (0..N).map(|i| i < 80 + extra).collect()
}

/// Two sealed suites; M2 against an identical incumbent (PATCH, tensor-identical).
fn incumbent() -> Vec<Suite> {
    vec![
        suite("humaneval", Some(above(0))),
        suite("mbpp", Some(above(0))),
    ]
}

fn sha(c: char) -> String {
    c.to_string().repeat(64)
}

/// A complete comparator block for an artifact arm (EXT-26).
pub(crate) fn block() -> Option<ComparatorBlock> {
    Some(ComparatorBlock {
        command: vec!["llama-cli".into(), "-m".into(), "stock.gguf".into()],
        version: "llama.cpp b6000".into(),
        env_sha256: "e".repeat(64),
        artifact_sha256: "f".repeat(64),
        log_path: "logs/stock.log".into(),
        image: None,
        started_utc: "2026-09-26T00:00:00Z".into(),
        finished_utc: "2026-09-26T00:10:00Z".into(),
    })
}

fn stock(humaneval: usize, mbpp: usize) -> Arm {
    Arm {
        identity: ArmIdentity::Artifact {
            name: "qwen3.5-4b-stock".into(),
            sha256: sha('a'),
        },
        role: ArmRole::Blocking,
        suites: vec![
            suite("humaneval", Some(above(humaneval))),
            suite("mbpp", Some(above(mbpp))),
        ],
        comparator: block(),
    }
}

fn api(role: ArmRole) -> Arm {
    Arm {
        identity: ArmIdentity::Api {
            model_id: "claude-haiku-4-5".into(),
            version: "20251001".into(),
        },
        role,
        // Far better than the candidate on both suites.
        suites: vec![
            suite("humaneval", Some(above(20))),
            suite("mbpp", Some(above(20))),
        ],
        comparator: None,
    }
}

fn verdict(class: ReleaseClass, inc: &[Suite], arms: &[Arm]) -> Verdict {
    gate(&PRE, class, inc, arms)
        .expect("comparable inputs")
        .verdict
}

#[test]
fn falsify_ext_023_below_stock_refused() {
    let patch = ReleaseClass::PatchTensorIdentical;
    // Control: M2 alone promotes, and so does M2 + a stock arm the candidate matches.
    assert_eq!(
        super::super::m2(&PRE, patch, &incumbent()).unwrap().verdict,
        Verdict::Promote
    );
    assert_eq!(
        verdict(patch, &incumbent(), &[stock(0, 0)]),
        Verdict::Promote
    );

    // Plant: stock passes 15 items on mbpp the candidate fails. McNemar one-sided
    // p = 2^-15, Holm-adjusted 2 * 2^-15 < 0.05, so the candidate is significantly
    // worse than stock on one sealed suite, and M2's promote is overruled.
    let r = gate(&PRE, patch, &incumbent(), &[stock(0, 15)]).unwrap();
    assert_eq!(
        r.m2.verdict,
        Verdict::Promote,
        "M2 alone would have promoted"
    );
    match &r.verdict {
        Verdict::Reject { reason } => {
            assert!(
                reason.contains("mbpp") && reason.contains("qwen3.5-4b-stock"),
                "{reason}"
            );
        }
        v => panic!("below stock must be refused, got {v}"),
    }
    let mbpp = &r.arms[0].suites[1];
    assert_eq!(mbpp.report.discordant, Some((0, 15)));
    assert!(mbpp
        .report
        .holm_worse
        .is_some_and(|h| h.rejected && (h.adjusted_p - 2.0 / 32768.0).abs() < 1e-12));

    // Non-inferior means not significantly worse: 3 stock-only passes (p = 1/8) is not.
    assert_eq!(
        verdict(patch, &incumbent(), &[stock(3, 3)]),
        Verdict::Promote
    );
}

#[test]
fn ext_30_first_release_records_stock_and_is_still_blocked() {
    let first = ReleaseClass::First;
    let inc = vec![suite("humaneval", None), suite("mbpp", None)];
    let r = gate(&PRE, first, &inc, &[stock(0, 5)]).unwrap();
    assert_eq!(r.verdict, Verdict::Promote);
    // The receipt records the stock levels beside the candidate's, and claims nothing.
    let levels: Vec<_> = r.arms[0]
        .suites
        .iter()
        .map(|s| (s.report.candidate_level, s.arm_level))
        .collect();
    assert_eq!(levels, vec![(0.8, 0.8), (0.8, 0.85)]);
    assert!(r.arms[0]
        .suites
        .iter()
        .all(|s| !s.report.holm_better.is_some_and(|h| h.rejected)));

    // The stock arm blocks a first release too.
    assert!(matches!(
        verdict(first, &inc, &[stock(15, 0)]),
        Verdict::Reject { .. }
    ));
}

#[test]
fn ext_30_report_only_arms_never_decide() {
    let patch = ReleaseClass::PatchTensorIdentical;
    // Haiku is significantly better than the candidate on both suites; it is reported
    // (Holm-rejected worse) and the verdict does not move.
    let r = gate(
        &PRE,
        patch,
        &incumbent(),
        &[stock(0, 0), api(ArmRole::ReportOnly)],
    )
    .unwrap();
    assert_eq!(r.verdict, Verdict::Promote);
    assert!(r.arms[1]
        .suites
        .iter()
        .all(|s| s.report.holm_worse.is_some_and(|h| h.rejected)));

    // A 9B control artifact that is report-only does not block either.
    let mut control = stock(20, 20);
    control.role = ArmRole::ReportOnly;
    control.identity = ArmIdentity::Artifact {
        name: "qwen3.5-9b".into(),
        sha256: sha('b'),
    };
    assert_eq!(
        verdict(patch, &incumbent(), &[stock(0, 0), control]),
        Verdict::Promote
    );
}

#[test]
fn ext_30_m2_verdict_is_kept_when_m2_does_not_promote() {
    // M2 rejects (the incumbent is 15 better on mbpp): stock cannot rescue it.
    let inc = vec![
        suite("humaneval", Some(above(0))),
        suite("mbpp", Some(above(15))),
    ];
    let v = verdict(ReleaseClass::PatchTensorIdentical, &inc, &[stock(0, 0)]);
    assert!(
        matches!(&v, Verdict::Reject { reason } if !reason.contains("stock")),
        "{v}"
    );
}

#[test]
fn ext_30_underpowered_against_stock_is_not_a_pass() {
    // Scored suite: identical to the incumbent (CI half-width 0), but against stock the
    // paired differences alternate +-2, so the CI is far wider than the 0.1 registered.
    let scored = |base: Vec<f64>| Suite {
        name: "judge".into(),
        data: SuiteData::Scored {
            candidate: vec![5.0; N],
            baseline: Some(base),
            higher_is_better: true,
            max_ci_half_width: 0.1,
        },
    };
    let inc = vec![scored(vec![5.0; N])];
    let noisy: Vec<f64> = (0..N).map(|i| if i % 2 == 0 { 3.0 } else { 7.0 }).collect();
    let arm = |base| Arm {
        identity: ArmIdentity::Artifact {
            name: "stock".into(),
            sha256: sha('c'),
        },
        role: ArmRole::Blocking,
        suites: vec![scored(base)],
        comparator: block(),
    };
    let patch = ReleaseClass::PatchTensorIdentical;
    assert!(matches!(
        verdict(patch, &inc, &[arm(noisy.clone())]),
        Verdict::Underpowered { ref suite, n_req } if suite == "judge" && n_req > N as u64
    ));
    // The receipt records a scored arm's level as its mean (50 x 3.0 + 50 x 7.0) / 100.
    let r = gate(&PRE, patch, &inc, &[arm(noisy)]).expect("comparable inputs");
    assert!((r.arms[0].suites[0].arm_level - 5.0).abs() < 1e-12);
    assert_eq!(verdict(patch, &inc, &[arm(vec![5.0; N])]), Verdict::Promote);
}

#[test]
fn ext_30_malformed_arms_are_refused_not_judged() {
    let patch = ReleaseClass::PatchTensorIdentical;
    let inc = incumbent();
    let refused = |arms: &[Arm]| gate(&PRE, patch, &inc, arms).unwrap_err();

    assert!(
        refused(&[]).contains("no blocking"),
        "the stock arm is mandatory"
    );
    assert!(refused(&[api(ArmRole::ReportOnly)]).contains("no blocking"));
    assert!(refused(&[stock(0, 0), api(ArmRole::Blocking)]).contains("non-hermetic"));

    for bad in [
        String::new(),
        "A".repeat(64),
        "a".repeat(63),
        "g".repeat(64),
    ] {
        let mut a = stock(0, 0);
        a.identity = ArmIdentity::Artifact {
            name: "s".into(),
            sha256: bad.clone(),
        };
        assert!(refused(&[a]).contains("sha256"), "{bad:?}");
    }
    let mut a = api(ArmRole::ReportOnly);
    a.identity = ArmIdentity::Api {
        model_id: "agy".into(),
        version: " ".into(),
    };
    assert!(refused(&[stock(0, 0), a]).contains("version"));

    let mut a = stock(0, 0);
    a.suites.pop();
    assert!(refused(&[a]).contains("sealed suites"), "a missing suite");
    let mut a = stock(0, 0);
    a.suites[1].name = "gsm8k".into();
    assert!(refused(&[a]).contains("sealed suites"), "a foreign suite");

    let mut a = stock(0, 0);
    if let SuiteData::Binary { candidate, .. } = &mut a.suites[0].data {
        candidate[0] = !candidate[0];
    }
    assert!(refused(&[a]).contains("candidate items"));

    let mut a = stock(0, 0);
    a.suites[0].data = SuiteData::Binary {
        candidate: cand(),
        baseline: None,
    };
    assert!(
        refused(&[a]).contains("qwen3.5-4b-stock"),
        "an arm always has a baseline"
    );
}

#[test]
fn falsify_ext_020_an_m2_artifact_arm_needs_a_complete_comparator_block() {
    let patch = ReleaseClass::PatchTensorIdentical;
    let inc = incumbent();
    let refused = |arm: Arm| gate(&PRE, patch, &inc, &[arm]).unwrap_err();

    let mut a = stock(0, 0);
    a.comparator = None;
    let e = refused(a);
    assert!(
        e.contains("FALSIFY-EXT-020") && e.contains("qwen3.5-4b-stock"),
        "{e}"
    );

    let plants: [(&str, fn(&mut ComparatorBlock)); 5] = [
        ("version", |b| b.version = " ".into()),
        ("command", |b| b.command.clear()),
        ("env_sha256", |b| b.env_sha256 = "E".repeat(64)),
        ("artifact_sha256", |b| b.artifact_sha256 = "f".repeat(63)),
        ("pinned by digest", |b| {
            b.image = Some("ghcr.io/x:latest".into())
        }),
    ];
    for (want, plant) in plants {
        let mut a = stock(0, 0);
        plant(a.comparator.as_mut().unwrap());
        let e = refused(a);
        assert!(
            e.contains(want) && e.contains("qwen3.5-4b-stock"),
            "{want}: {e}"
        );
    }

    // A report-only API arm has no artifact to hash and needs no block.
    assert_eq!(
        verdict(patch, &inc, &[stock(0, 0), api(ArmRole::ReportOnly)]),
        Verdict::Promote
    );
}
