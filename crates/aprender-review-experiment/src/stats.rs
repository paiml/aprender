//! REX-001 frozen analysis code (spec §2.3, §3, §5.4).
//!
//! This file is part of the pre-registration bundle (`prereg.rs`): its sha256 is
//! folded into the prereg sha that every receipt carries. Changing a single byte
//! here after REX-00 is a new spec version (R-1), and data gathered before it is
//! `exploratory`.
//!
//! Pre-registered choices, stated once so nobody re-derives them:
//! - Wilson score interval, z = 1.959963984540054 (two-sided 95 %).
//! - McNemar EXACT, paired, one-sided: under H0 the discordant count `b`
//!   (challenger right, champion wrong) is Binomial(b + c, 1/2); p = P(X >= b).
//! - Bootstrap: percentile CI over items resampled with replacement, 10 000
//!   resamples, SplitMix64 seeded (the generator is written out here so its
//!   stream can never change under a dependency bump). One-sided bootstrap p =
//!   fraction of resamples whose difference is < 0.
//! - v2 H4: the bootstrap statistic is `p(apr) − min(p(Haiku), p(agy))` with
//!   the min computed inside each resample (`bootstrap_vs_min_voter`). The
//!   decision is the Holm-adjusted one-sided p; the 95 % CI is report-only.
//! - v2 H5: recall on class R only; class-P recall and precision are
//!   descriptive (mutant confound).
//! - v2 descriptive: Cohen's κ on per-item errors between apr and each voter
//!   (`error_kappa`); a κ rise with no recall gain is an andon (`kappa_andon`).
//! - Holm step-down over the family H1–H6 at overall alpha = 0.05. The count
//!   and threshold tests (H1, H2, H6) enter the family as p = 0 when rejected
//!   and p = 1 when not: they are deterministic rules, not sampling statistics.
//! - Percentiles: nearest-rank (`ceil(q * n)`-th order statistic).
//! - H6 noise = 2 x sample sd (n - 1) of the alone runs; rejected when
//!   mean(co-located) - mean(alone) > noise.

/// z for a two-sided 95 % interval.
pub const Z95: f64 = 1.959_963_984_540_054;
/// Overall family-wise alpha for H1–H6 (spec §3).
pub const ALPHA: f64 = 0.05;
/// Bootstrap resamples (spec §3 H4).
pub const BOOTSTRAP_RESAMPLES: usize = 10_000;

/// Wilson score interval for `k` successes in `n` trials. `None` when `n == 0`.
#[must_use]
pub fn wilson(k: u64, n: u64, z: f64) -> Option<(f64, f64)> {
    if n == 0 || k > n {
        return None;
    }
    let n_f = n as f64;
    let p = k as f64 / n_f;
    let z2 = z * z;
    let denom = 1.0 + z2 / n_f;
    let centre = (p + z2 / (2.0 * n_f)) / denom;
    let half = z * ((p * (1.0 - p) / n_f) + z2 / (4.0 * n_f * n_f)).sqrt() / denom;
    Some(((centre - half).max(0.0), (centre + half).min(1.0)))
}

/// Half-width of the Wilson interval — the §2.2 sample-size rule reads this.
#[must_use]
pub fn wilson_half_width(k: u64, n: u64) -> Option<f64> {
    wilson(k, n, Z95).map(|(lo, hi)| (hi - lo) / 2.0)
}

fn ln_choose(n: u64, k: u64) -> f64 {
    // Sum of logs is exact enough for n <= a few thousand and has no overflow.
    let k = k.min(n - k);
    (0..k)
        .map(|i| ((n - i) as f64).ln() - ((i + 1) as f64).ln())
        .sum()
}

/// McNemar exact, paired, one-sided. `b` = pairs where the challenger is right
/// and the champion wrong; `c` = the reverse. Returns P(X >= b), X ~ Bin(b+c, ½).
/// With no discordant pairs there is no evidence: p = 1.
#[must_use]
pub fn mcnemar_exact_one_sided(b: u64, c: u64) -> f64 {
    let n = b + c;
    if n == 0 {
        return 1.0;
    }
    let ln_half_n = (n as f64) * 0.5_f64.ln();
    let p: f64 = (b..=n).map(|x| (ln_choose(n, x) + ln_half_n).exp()).sum();
    p.min(1.0)
}

/// SplitMix64 — the pre-registered bootstrap generator.
#[derive(Debug, Clone)]
pub struct SplitMix64(u64);

impl SplitMix64 {
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform index in `0..n` (Lemire multiply-shift; n > 0).
    pub fn below(&mut self, n: usize) -> usize {
        ((u128::from(self.next_u64()) * n as u128) >> 64) as usize
    }
}

/// One paired review item for two lanes, A (the candidate) and B (the baseline).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Paired {
    /// Ground truth: the item carries a defect (class P or R).
    pub defect: bool,
    /// Lane A said FAIL (a parsed FAIL; `Unparsed`/`NotRun` are NOT a FAIL).
    pub a_fail: bool,
    /// Lane B said FAIL.
    pub b_fail: bool,
}

/// Which lane-level rate a bootstrap compares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rate {
    /// FAIL on defect ÷ all FAIL.
    Precision,
    /// FAIL on defect ÷ defects.
    Recall,
}

fn rate(items: &[Paired], idx: &[usize], lane_a: bool, which: Rate) -> Option<f64> {
    let (mut num, mut den) = (0u64, 0u64);
    for &i in idx {
        let it = items[i];
        let fail = if lane_a { it.a_fail } else { it.b_fail };
        match which {
            Rate::Precision => {
                if fail {
                    den += 1;
                    num += u64::from(it.defect);
                }
            }
            Rate::Recall => {
                if it.defect {
                    den += 1;
                    num += u64::from(fail);
                }
            }
        }
    }
    (den > 0).then(|| num as f64 / den as f64)
}

/// Result of a paired bootstrap of `rate(A) - rate(B)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BootstrapDiff {
    /// Point estimate on the full sample.
    pub estimate: f64,
    /// Percentile CI at `1 - alpha` two-sided.
    pub ci: (f64, f64),
    /// One-sided p: fraction of defined resamples with difference < 0.
    pub p_below_zero: f64,
    /// Resamples where either rate was undefined (empty denominator); excluded.
    pub undefined: usize,
}

/// Paired bootstrap of `rate(A) - rate(B)` over items (spec §3 H4/H5).
/// `None` when the point estimate is undefined or no resample is defined.
#[must_use]
pub fn paired_bootstrap_diff(
    items: &[Paired],
    which: Rate,
    resamples: usize,
    seed: u64,
    alpha: f64,
) -> Option<BootstrapDiff> {
    let all: Vec<usize> = (0..items.len()).collect();
    let estimate = rate(items, &all, true, which)? - rate(items, &all, false, which)?;
    let mut rng = SplitMix64::new(seed);
    let mut diffs = Vec::with_capacity(resamples);
    let mut undefined = 0usize;
    let mut idx = vec![0usize; items.len()];
    for _ in 0..resamples {
        for slot in &mut idx {
            *slot = rng.below(items.len());
        }
        match (
            rate(items, &idx, true, which),
            rate(items, &idx, false, which),
        ) {
            (Some(a), Some(b)) => diffs.push(a - b),
            _ => undefined += 1,
        }
    }
    if diffs.is_empty() {
        return None;
    }
    diffs.sort_by(f64::total_cmp);
    let below = diffs.iter().filter(|d| **d < 0.0).count();
    Some(BootstrapDiff {
        estimate,
        ci: (
            percentile_sorted(&diffs, alpha / 2.0),
            percentile_sorted(&diffs, 1.0 - alpha / 2.0),
        ),
        p_below_zero: below as f64 / diffs.len() as f64,
        undefined,
    })
}

/// One H4 item scored by apr and the two existing voters (Haiku, agy).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VsVoters {
    /// Ground truth: the item carries a defect (class P or R).
    pub defect: bool,
    /// apr said FAIL.
    pub apr_fail: bool,
    /// The two voting lanes said FAIL, in a fixed order (Haiku, agy).
    pub voter_fail: [bool; 2],
}

fn rate_of(items: &[VsVoters], idx: &[usize], lane: Option<usize>, which: Rate) -> Option<f64> {
    let pick = |it: &VsVoters| lane.map_or(it.apr_fail, |v| it.voter_fail[v]);
    let (mut num, mut den) = (0u64, 0u64);
    for &i in idx {
        let it = &items[i];
        let fail = pick(it);
        match which {
            Rate::Precision if fail => {
                den += 1;
                num += u64::from(it.defect);
            }
            Rate::Recall if it.defect => {
                den += 1;
                num += u64::from(fail);
            }
            _ => {}
        }
    }
    (den > 0).then(|| num as f64 / den as f64)
}

/// `rate(apr) − min(rate(Haiku), rate(agy))` on one index set.
fn diff_vs_min(items: &[VsVoters], idx: &[usize], which: Rate) -> Option<f64> {
    let a = rate_of(items, idx, None, which)?;
    let lo = rate_of(items, idx, Some(0), which)?.min(rate_of(items, idx, Some(1), which)?);
    Some(a - lo)
}

/// H4 (spec v2 §3): paired bootstrap of `p(apr) − min(p(Haiku), p(agy))`.
/// The min is taken INSIDE each resample, so the weaker voter is chosen on
/// that resample and the selection is part of the sampling distribution. The
/// decision is the Holm-adjusted `p_below_zero`; the CI is report-only.
#[must_use]
pub fn bootstrap_vs_min_voter(
    items: &[VsVoters],
    which: Rate,
    resamples: usize,
    seed: u64,
    alpha: f64,
) -> Option<BootstrapDiff> {
    let all: Vec<usize> = (0..items.len()).collect();
    let estimate = diff_vs_min(items, &all, which)?;
    let mut rng = SplitMix64::new(seed);
    let mut diffs = Vec::with_capacity(resamples);
    let mut undefined = 0usize;
    let mut idx = vec![0usize; items.len()];
    for _ in 0..resamples {
        for slot in &mut idx {
            *slot = rng.below(items.len());
        }
        match diff_vs_min(items, &idx, which) {
            Some(d) => diffs.push(d),
            None => undefined += 1,
        }
    }
    if diffs.is_empty() {
        return None;
    }
    diffs.sort_by(f64::total_cmp);
    let below = diffs.iter().filter(|d| **d < 0.0).count();
    Some(BootstrapDiff {
        estimate,
        ci: (
            percentile_sorted(&diffs, alpha / 2.0),
            percentile_sorted(&diffs, 1.0 - alpha / 2.0),
        ),
        p_below_zero: below as f64 / diffs.len() as f64,
        undefined,
    })
}

/// Cohen's κ between two lanes' per-item ERROR indicators (spec v2 §3,
/// descriptive): 1 = they miss the same items, 0 = overlap at chance.
/// `None` when the lengths differ, the sample is empty, or chance agreement
/// is 1 (both lanes constant), where κ is undefined.
#[must_use]
pub fn error_kappa(a_err: &[bool], b_err: &[bool]) -> Option<f64> {
    if a_err.len() != b_err.len() || a_err.is_empty() {
        return None;
    }
    let n = a_err.len() as f64;
    let agree = a_err.iter().zip(b_err).filter(|(x, y)| x == y).count() as f64 / n;
    let pa = a_err.iter().filter(|x| **x).count() as f64 / n;
    let pb = b_err.iter().filter(|x| **x).count() as f64 / n;
    let chance = pa * pb + (1.0 - pa) * (1.0 - pb);
    (chance < 1.0).then(|| (agree - chance) / (1.0 - chance))
}

/// Spec v2 §3 andon: error overlap with a voter rose across a silver-label
/// training step while recall did not rise. Undefined inputs are no andon.
#[must_use]
pub fn kappa_andon(kappa: (Option<f64>, Option<f64>), recall: (f64, f64)) -> bool {
    matches!(kappa, (Some(before), Some(after)) if after > before) && recall.1 <= recall.0
}

/// Nearest-rank percentile of an ascending slice. `q` in (0, 1].
#[must_use]
pub fn percentile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let rank = (q * sorted.len() as f64).ceil().max(1.0) as usize;
    sorted[rank.min(sorted.len()) - 1]
}

/// Nearest-rank percentile of an unsorted sample.
#[must_use]
pub fn percentile(sample: &[f64], q: f64) -> f64 {
    let mut s = sample.to_vec();
    s.sort_by(f64::total_cmp);
    percentile_sorted(&s, q)
}

/// Holm step-down. Returns (adjusted p, rejected) in input order.
#[must_use]
pub fn holm(p: &[f64], alpha: f64) -> Vec<(f64, bool)> {
    let m = p.len();
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| p[a].total_cmp(&p[b]).then(a.cmp(&b)));
    let mut out = vec![(1.0, false); m];
    let mut running = 0.0_f64;
    let mut still_rejecting = true;
    for (rank, &i) in order.iter().enumerate() {
        let factor = (m - rank) as f64;
        running = running.max((p[i] * factor).min(1.0));
        still_rejecting = still_rejecting && p[i] <= alpha / factor;
        out[i] = (running, still_rejecting);
    }
    out
}

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

/// Sample standard deviation (n - 1). `None` for fewer than two values.
#[must_use]
pub fn sample_sd(x: &[f64]) -> Option<f64> {
    if x.len() < 2 {
        return None;
    }
    let m = mean(x);
    Some((x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (x.len() - 1) as f64).sqrt())
}

/// H6 interference verdict for one cell.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interference {
    /// mean(co-located) − mean(alone), in the workload's unit (seconds).
    pub slowdown: f64,
    /// 2 × sd(alone).
    pub noise: f64,
    /// slowdown > noise.
    pub rejected: bool,
}

/// H6: reference workload wall-clock alone vs co-located with the serve.
#[must_use]
pub fn interference(alone: &[f64], colocated: &[f64]) -> Option<Interference> {
    let noise = 2.0 * sample_sd(alone)?;
    if colocated.is_empty() {
        return None;
    }
    let slowdown = mean(colocated) - mean(alone);
    Some(Interference {
        slowdown,
        noise,
        rejected: slowdown > noise,
    })
}

/// H1/H2 are counts: any divergent item rejects. Enters Holm as p ∈ {0, 1}.
#[must_use]
pub fn count_rule_p(divergent: u64) -> f64 {
    if divergent > 0 {
        0.0
    } else {
        1.0
    }
}

/// McNemar discordant counts from per-item correctness of (challenger, champion).
#[must_use]
pub fn discordant(pairs: &[(bool, bool)]) -> (u64, u64) {
    pairs
        .iter()
        .fold((0, 0), |(b, c), &(ch, cp)| match (ch, cp) {
            (true, false) => (b + 1, c),
            (false, true) => (b, c + 1),
            _ => (b, c),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn wilson_matches_hand_computed() {
        // k=35, n=70, p=.5: centre .5, half = 1.96*sqrt(.25/70 + 3.8415/19600)/1.05488 = .114044
        let (lo, hi) = wilson(35, 70, Z95).expect("defined");
        assert!(close(lo, 0.385_957, 1e-5), "{lo}");
        assert!(close(hi, 0.614_043, 1e-5), "{hi}");
        // §2.2 basis: 70 defect items at p = .5. The spec's "≈ 0.117" is the WALD
        // half-width 1.96*sqrt(.25/70); Wilson, the pre-registered interval, gives 0.114.
        let hw = wilson_half_width(35, 70).expect("defined");
        assert!(close(hw, 0.114_043, 1e-5), "{hw}");
        assert_eq!(wilson(0, 0, Z95), None);
        let (lo0, _) = wilson(0, 10, Z95).expect("defined");
        assert!(close(lo0, 0.0, 1e-12));
    }

    #[test]
    fn mcnemar_exact_matches_binomial_tail() {
        // b=8, c=2: P(X>=8 | n=10) = (45+10+1)/1024 = 0.0546875
        assert!(close(mcnemar_exact_one_sided(8, 2), 56.0 / 1024.0, 1e-12));
        // b=5, c=0: 1/32
        assert!(close(mcnemar_exact_one_sided(5, 0), 1.0 / 32.0, 1e-12));
        // b=0: whole distribution
        assert!(close(mcnemar_exact_one_sided(0, 7), 1.0, 1e-12));
        assert!(close(mcnemar_exact_one_sided(0, 0), 1.0, 0.0));
        assert_eq!(
            discordant(&[(true, false), (true, false), (false, true), (true, true)]),
            (2, 1)
        );
    }

    #[test]
    fn holm_matches_hand_computed() {
        // p sorted .01 .02 .03 .5 ; thresholds .0125 .0167 .025 .05 -> reject .01 only
        let r = holm(&[0.03, 0.01, 0.5, 0.02], 0.05);
        assert_eq!(
            r.iter().map(|x| x.1).collect::<Vec<_>>(),
            vec![false, true, false, false]
        );
        assert!(close(r[1].0, 0.04, 1e-12));
        assert!(close(r[3].0, 0.06, 1e-12));
        assert!(close(r[0].0, 0.06, 1e-12)); // monotone: max(0.06, 0.03*2)
        assert!(close(r[2].0, 0.5, 1e-12));
        // deterministic count tests enter as 0 / 1
        let r = holm(&[count_rule_p(3), count_rule_p(0)], 0.05);
        assert_eq!((r[0].1, r[1].1), (true, false));
    }

    #[test]
    fn percentile_nearest_rank() {
        let s: Vec<f64> = (1..=20).map(f64::from).collect();
        assert!(close(percentile(&s, 0.95), 19.0, 0.0));
        assert!(close(percentile(&s, 0.5), 10.0, 0.0));
        assert!(percentile(&[], 0.5).is_nan());
    }

    #[test]
    fn splitmix_stream_is_frozen() {
        // Reference values of SplitMix64 seeded 0 (Vigna's published sequence).
        let mut r = SplitMix64::new(0);
        assert_eq!(r.next_u64(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(r.next_u64(), 0x6E78_9E6A_A1B9_65F4);
    }

    #[test]
    fn bootstrap_is_seeded_and_signed() {
        // A strictly better lane A: identical except A catches 5 more defects.
        let mut items = Vec::new();
        for i in 0..40 {
            let defect = i < 20;
            let b_fail = defect && i < 10;
            let a_fail = defect && i < 15;
            items.push(Paired {
                defect,
                a_fail,
                b_fail,
            });
        }
        let x = paired_bootstrap_diff(&items, Rate::Recall, 2000, 7, ALPHA).expect("defined");
        let y = paired_bootstrap_diff(&items, Rate::Recall, 2000, 7, ALPHA).expect("defined");
        assert_eq!(x, y, "same seed must give the same bootstrap");
        assert!(close(x.estimate, 0.25, 1e-12));
        assert!(x.ci.0 >= 0.0 && x.p_below_zero == 0.0, "{x:?}");
        // Precision: A and B both 1.0 -> diff 0 everywhere.
        let p = paired_bootstrap_diff(&items, Rate::Precision, 500, 7, ALPHA).expect("defined");
        assert!(close(p.estimate, 0.0, 0.0) && close(p.ci.0, 0.0, 0.0));
    }

    /// Three lanes, equal point precision 10/12, false positives on
    /// disjoint good items: which voter is weaker flips between resamples.
    fn tied_voters() -> Vec<VsVoters> {
        (0..40)
            .map(|i| {
                let defect = i < 20;
                let caught = i < 10;
                VsVoters {
                    defect,
                    apr_fail: caught || i == 20 || i == 21,
                    voter_fail: [caught || i == 22 || i == 23, caught || i == 24 || i == 25],
                }
            })
            .collect()
    }

    #[test]
    fn falsify_h4_min_is_taken_inside_each_resample() {
        let items = tied_voters();
        let inside =
            bootstrap_vs_min_voter(&items, Rate::Precision, 2000, 4354, ALPHA).expect("defined");
        // The v1 rule: pick the weaker voter ONCE on the full sample (a tie →
        // voter 0), then bootstrap against it. Same seed → same resamples.
        let fixed: Vec<Paired> = items
            .iter()
            .map(|t| Paired {
                defect: t.defect,
                a_fail: t.apr_fail,
                b_fail: t.voter_fail[0],
            })
            .collect();
        let once =
            paired_bootstrap_diff(&fixed, Rate::Precision, 2000, 4354, ALPHA).expect("defined");
        assert!(close(inside.estimate, 0.0, 1e-12) && close(once.estimate, 0.0, 1e-12));
        // a − min(v0, v1) ≥ a − v0 on every resample, strictly on those where
        // v1 is the weaker: the two rules must give different p.
        assert!(
            inside.p_below_zero < once.p_below_zero,
            "{inside:?} vs {once:?}"
        );
        // Voter order is irrelevant under min().
        let swapped: Vec<VsVoters> = items
            .iter()
            .map(|t| VsVoters {
                voter_fail: [t.voter_fail[1], t.voter_fail[0]],
                ..*t
            })
            .collect();
        assert_eq!(
            bootstrap_vs_min_voter(&swapped, Rate::Precision, 2000, 4354, ALPHA),
            Some(inside)
        );
    }

    #[test]
    fn error_kappa_matches_hand_computed() {
        // agree 3/4, pa = .5, pb = .25, chance = .5 → κ = .5
        let k = error_kappa(&[true, true, false, false], &[true, false, false, false]);
        assert!(close(k.expect("defined"), 0.5, 1e-12));
        assert_eq!(error_kappa(&[true, false], &[true, false]), Some(1.0));
        assert_eq!(
            error_kappa(&[false, false], &[false, false]),
            None,
            "chance = 1"
        );
        assert_eq!(
            error_kappa(&[true], &[true, false]),
            None,
            "length mismatch"
        );
        assert_eq!(error_kappa(&[], &[]), None);
    }

    #[test]
    fn kappa_andon_fires_only_on_overlap_rise_without_recall_gain() {
        assert!(kappa_andon((Some(0.2), Some(0.4)), (0.6, 0.6)));
        assert!(kappa_andon((Some(0.2), Some(0.4)), (0.6, 0.5)));
        assert!(
            !kappa_andon((Some(0.2), Some(0.4)), (0.6, 0.7)),
            "recall gained"
        );
        assert!(!kappa_andon((Some(0.4), Some(0.4)), (0.6, 0.6)), "no rise");
        assert!(
            !kappa_andon((None, Some(0.9)), (0.6, 0.6)),
            "undefined before"
        );
    }

    #[test]
    fn interference_rule() {
        let alone = [10.0, 10.2, 9.8, 10.1, 9.9];
        let sd = sample_sd(&alone).expect("n>=2");
        assert!(close(sd, 0.158_113_9, 1e-6));
        let quiet = interference(&alone, &[10.1, 10.2, 10.0, 10.2, 10.1]).expect("defined");
        assert!(!quiet.rejected, "{quiet:?}");
        let loud = interference(&alone, &[11.0, 11.2, 10.9, 11.1, 11.0]).expect("defined");
        assert!(loud.rejected, "{loud:?}");
        assert_eq!(interference(&[1.0], &[2.0]), None);
    }
}
