//! EXT-12 (aprender#4394): M2 statistics against hand-computed fixtures.

use super::*;

const PRE: M2Prereg = M2Prereg::REX_001;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-12
}

/// A binary suite with `b` candidate-only passes, `c` baseline-only passes, and
/// `both`/`neither` concordant pairs.
fn bin(name: &str, b: usize, c: usize, both: usize, neither: usize) -> Suite {
    let mut cand = Vec::new();
    let mut base = Vec::new();
    for (n, x, y) in [
        (b, true, false),
        (c, false, true),
        (both, true, true),
        (neither, false, false),
    ] {
        cand.extend(std::iter::repeat(x).take(n));
        base.extend(std::iter::repeat(y).take(n));
    }
    Suite {
        name: name.into(),
        data: SuiteData::Binary {
            candidate: cand,
            baseline: Some(base),
        },
    }
}

fn scored(name: &str, cand: Vec<f64>, base: Vec<f64>, higher_is_better: bool, max: f64) -> Suite {
    Suite {
        name: name.into(),
        data: SuiteData::Scored {
            candidate: cand,
            baseline: Some(base),
            higher_is_better,
            max_ci_half_width: max,
        },
    }
}

fn verdict(class: ReleaseClass, suites: &[Suite]) -> Verdict {
    m2(&PRE, class, suites).unwrap().verdict
}

#[test]
fn ext_12_mcnemar_matches_hand_binomial_tails() {
    // P(X ≥ b), X ~ Bin(b + c, ½), summed by hand.
    assert!(
        close(mcnemar_exact_one_sided(0, 0), 1.0),
        "no discordant pairs: no evidence"
    );
    assert!(close(mcnemar_exact_one_sided(3, 0), 1.0 / 8.0));
    assert!(close(mcnemar_exact_one_sided(5, 0), 1.0 / 32.0));
    assert!(close(mcnemar_exact_one_sided(6, 0), 1.0 / 64.0));
    assert!(close(mcnemar_exact_one_sided(5, 1), 7.0 / 64.0)); // (6 + 1) / 2⁶
    assert!(close(mcnemar_exact_one_sided(3, 3), 42.0 / 64.0)); // (20 + 15 + 6 + 1) / 2⁶
    assert!(close(mcnemar_exact_one_sided(0, 6), 1.0));
    assert!(close(mcnemar_exact_one_sided(12, 2), 106.0 / 16384.0)); // (91 + 14 + 1) / 2¹⁴
}

#[test]
fn ext_12_wilson_and_the_sample_size_rule_match_hand_values() {
    let z = PRE.z;
    // p = 0 or 1: half-width is z² / (2(n + z²)).
    let edge = |n: f64| z * z / (2.0 * (n + z * z));
    assert!(close(wilson_half_width(10, 10, z).unwrap(), edge(10.0)));
    assert!(close(wilson_half_width(0, 13, z).unwrap(), edge(13.0)));
    assert!((wilson_half_width(50, 100, z).unwrap() - 0.096_168_469_634_004_4).abs() < 1e-12);
    assert_eq!(wilson_half_width(1, 0, z), None);
    // n + z² ≥ z²/0.24 at p = 1 gives n ≥ 12.16, so 13.
    assert_eq!(wilson_n_req(10, 10, z, 0.12), 13);
    assert_eq!(wilson_n_req(5, 10, z, 0.12), 63);
    assert_eq!(wilson_n_req(9, 10, z, 0.12), 24);
}

#[test]
fn ext_12_holm_matches_a_hand_step_down() {
    // Sorted: 0.01 (×3 = 0.03, ≤ 0.05/3 → reject), 0.03 (×2 = 0.06, > 0.05/2 → stop), 0.04.
    let h = holm(&[0.01, 0.04, 0.03], 0.05);
    let got: Vec<(f64, bool)> = h.iter().map(|x| (x.adjusted_p, x.rejected)).collect();
    assert_eq!(got.len(), 3);
    assert!(close(got[0].0, 0.03) && got[0].1);
    assert!(
        close(got[1].0, 0.06) && !got[1].1,
        "running max carries 0.06 onto 0.04"
    );
    assert!(close(got[2].0, 0.06) && !got[2].1);
    // Once one fails, larger p never rejects even if it would alone.
    assert!(
        holm(&[0.03, 0.04], 0.05).iter().all(|x| !x.rejected),
        "stopped at 0.03 > 0.025"
    );
    assert!(
        holm(&[0.025, 0.05], 0.05).iter().all(|x| x.rejected),
        "p ≤ α/factor is inclusive"
    );
}

#[test]
fn ext_12_splitmix64_is_the_reference_generator() {
    // The published SplitMix64 outputs for seed 0.
    let mut r = SplitMix64::new(0);
    assert_eq!(r.next_u64(), 0xe220_a839_7b1d_cdaf);
    assert_eq!(r.next_u64(), 0x6e78_9e6a_a1b9_65f4);
    assert_eq!(r.next_u64(), 0x06c4_5d18_8009_454f);
    let mut r = SplitMix64::new(4354);
    assert!((0..1000).all(|_| r.below(7) < 7));
}

#[test]
fn ext_12_bootstrap_fixtures() {
    assert!(
        close(percentile_sorted(&[1.0, 2.0, 3.0, 4.0], 0.025), 1.0),
        "rank floors at 1"
    );
    assert!(close(percentile_sorted(&[1.0, 2.0, 3.0, 4.0], 0.5), 2.0));
    assert!(close(percentile_sorted(&[1.0, 2.0, 3.0, 4.0], 0.975), 4.0));
    // A constant difference resamples to itself.
    let b = bootstrap_mean(&[0.5; 20], 4354, 1000);
    assert_eq!(b.ci, (0.5, 0.5));
    assert!(close(b.share_le_zero, 0.0) && close(b.share_ge_zero, 1.0));
    // Identical arms: neither direction has evidence.
    let b = bootstrap_mean(&[0.0; 20], 4354, 1000);
    assert!(close(b.share_le_zero, 1.0) && close(b.share_ge_zero, 1.0));
    // Seeded: the same input gives the same CI.
    let d: Vec<f64> = (0..50).map(|i| f64::from(i % 7) - 3.0).collect();
    assert_eq!(
        bootstrap_mean(&d, 4354, 500).ci,
        bootstrap_mean(&d, 4354, 500).ci
    );
}

#[test]
fn ext_12_verdicts_follow_section_3_6() {
    use ReleaseClass::*;
    // 12 vs 2 discordant over 100 items: p = 106/16384 ≈ 0.0065 → better.
    let better = bin("humaneval", 12, 2, 76, 10);
    assert_eq!(verdict(Improvement, &[better.clone()]), Verdict::Promote);
    // The mirror image is significantly worse, whatever the class.
    let worse = bin("humaneval", 2, 12, 76, 10);
    for class in [Improvement, PatchTensorIdentical] {
        let v = verdict(class, &[worse.clone()]);
        assert!(
            matches!(&v, Verdict::Reject { reason } if reason.contains("humaneval")),
            "{v:?}"
        );
    }
    // One better suite does not outvote a worse one.
    let mbpp = bin("mbpp", 2, 12, 76, 10);
    assert!(matches!(
        verdict(Improvement, &[better.clone(), mbpp]),
        Verdict::Reject { .. }
    ));

    // Identical arms: no evidence either way. Only PATCH (tensor-identical) promotes.
    let same = bin("humaneval", 0, 0, 80, 20);
    assert!(matches!(
        verdict(Improvement, &[same.clone()]),
        Verdict::Reject { .. }
    ));
    assert_eq!(verdict(PatchTensorIdentical, &[same]), Verdict::Promote);

    // Holm across suites: p = 1/32 alone passes 0.05, beside a null suite it needs 0.025.
    let a = bin("a", 5, 0, 85, 10);
    let null = bin("b", 3, 3, 84, 10);
    assert_eq!(verdict(Improvement, &[a.clone()]), Verdict::Promote);
    assert!(matches!(
        verdict(Improvement, &[a, null]),
        Verdict::Reject { .. }
    ));

    // First release: no incumbent, absolute levels, promotes when powered.
    let first = Suite {
        name: "humaneval".into(),
        data: SuiteData::Binary {
            candidate: vec![true; 60].into_iter().chain([false; 40]).collect(),
            baseline: None,
        },
    };
    let r = m2(&PRE, First, &[first]).unwrap();
    assert_eq!(r.verdict, Verdict::Promote);
    assert!(close(r.suites[0].candidate_level, 0.6));
    assert_eq!(
        (r.suites[0].p_better, r.suites[0].holm_worse),
        (None, None),
        "no comparison claimed"
    );
}

#[test]
fn ext_12_scored_suites_use_the_oriented_bootstrap() {
    use ReleaseClass::*;
    let base: Vec<f64> = (0..40).map(|i| 8.0 + f64::from(i % 5) * 0.25).collect();
    let lower: Vec<f64> = base.iter().map(|x| x - 0.5).collect();
    // Perplexity: lower is better.
    let r = m2(
        &PRE,
        Improvement,
        &[scored("ppl", lower.clone(), base.clone(), false, 0.2)],
    )
    .unwrap();
    assert_eq!(r.verdict, Verdict::Promote);
    assert_eq!(
        r.suites[0].diff_ci,
        Some((0.5, 0.5)),
        "oriented so better is positive"
    );
    // The same numbers on a higher-is-better suite are a regression.
    let v = verdict(Improvement, &[scored("score", lower, base, true, 0.2)]);
    assert!(matches!(v, Verdict::Reject { reason } if reason.contains("score")));
}

#[test]
fn falsify_ext_015_underpowered_m2_refuses_promotion() {
    use ReleaseClass::*;
    // Candidate right on all 10 items, incumbent wrong on all: p = 1/1024, overwhelming.
    // But the Wilson half-width at 10/10 is 0.1388 > 0.12, so M2 must not promote.
    let planted = bin("humaneval", 10, 0, 0, 0);
    let r = m2(&PRE, Improvement, &[planted]).unwrap();
    assert!(
        r.suites[0].holm_better.unwrap().rejected,
        "the comparison itself is significant"
    );
    assert_eq!(
        r.verdict,
        Verdict::Underpowered {
            n_req: 13,
            suite: "humaneval".into()
        },
        "an underpowered comparison promoted"
    );
    assert_eq!(r.verdict.to_string(), "underpowered(13)");
    // No release class escapes the rule.
    for class in [PatchTensorIdentical, Improvement] {
        let v = verdict(class, &[bin("humaneval", 0, 0, 10, 0)]);
        assert!(
            matches!(v, Verdict::Underpowered { .. }),
            "{class:?}: {v:?}"
        );
    }
    let first = Suite {
        name: "humaneval".into(),
        data: SuiteData::Binary {
            candidate: vec![true; 10],
            baseline: None,
        },
    };
    assert!(matches!(
        verdict(First, &[first]),
        Verdict::Underpowered { n_req: 13, .. }
    ));
    // Scored suites: a CI wider than the pre-registered rule is underpowered too.
    let noisy_c: Vec<f64> = (0..8)
        .map(|i| if i % 2 == 0 { 3.0 } else { -1.0 })
        .collect();
    let v = verdict(
        Improvement,
        &[scored("ppl", noisy_c, vec![0.0; 8], true, 0.1)],
    );
    assert!(matches!(v, Verdict::Underpowered { .. }), "{v:?}");

    // Control: 13 of 13 meets the rule (half-width 0.1140) and promotes.
    assert_eq!(
        verdict(Improvement, &[bin("humaneval", 13, 0, 0, 0)]),
        Verdict::Promote
    );
    assert_eq!(Verdict::Promote.to_string(), "promote");
}

#[test]
fn ext_12_incomparable_suites_are_refused_not_judged() {
    use ReleaseClass::*;
    let ok = bin("a", 1, 0, 99, 0);
    assert!(m2(&PRE, Improvement, &[]).is_err(), "no suites");
    assert!(
        m2(&PRE, Improvement, &[ok.clone(), ok.clone()]).is_err(),
        "duplicate name"
    );
    assert!(
        m2(&PRE, First, &[ok]).is_err(),
        "a first release has no incumbent"
    );
    let no_base = Suite {
        name: "a".into(),
        data: SuiteData::Binary {
            candidate: vec![true],
            baseline: None,
        },
    };
    assert!(m2(&PRE, Improvement, &[no_base]).is_err(), "no baseline");
    let short = Suite {
        name: "a".into(),
        data: SuiteData::Binary {
            candidate: vec![true; 3],
            baseline: Some(vec![true; 2]),
        },
    };
    assert!(m2(&PRE, Improvement, &[short]).is_err(), "unpaired items");
    let empty = Suite {
        name: "a".into(),
        data: SuiteData::Binary {
            candidate: vec![],
            baseline: Some(vec![]),
        },
    };
    assert!(m2(&PRE, Improvement, &[empty]).is_err(), "no items");
    assert!(m2(
        &PRE,
        Improvement,
        &[scored("s", vec![f64::NAN], vec![0.0], true, 1.0)]
    )
    .is_err());
    assert!(
        m2(
            &PRE,
            Improvement,
            &[scored("s", vec![1.0], vec![0.0], true, 0.0)]
        )
        .is_err(),
        "no rule"
    );
}
