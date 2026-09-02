//! PP-LLAMA-001 v3.0 §4.3 — the **replicate-unit** estimator.
//!
//! Window statistics (`agg`, `prefill`, `vram_peak`) are one number per band
//! run, so there is nothing inside a band to resample: the unit of variation is
//! the *replicate*. §4.3 decides them with
//!
//! > the mean of the per-replicate `ln(x_apr / x_llama)`, bounded below by a
//! > one-sided 95% Student-t bound with `df = n − 1`, exponentiated.
//!
//! Three things about that are load-bearing and each is a separate test here.
//!
//! **Logs, not raw ratios.** A ratio's sampling distribution is skewed and its
//! arithmetic mean is not the ratio of the means. `ln` makes the paired
//! comparison additive, and exponentiating the bound returns a bound on the
//! ratio.
//!
//! **One-sided, not two-sided.** P-5 asks "is the lower bound at or above
//! `1 − δ`", which is a one-sided question. The two-tailed table
//! (`llm/benchmark.rs::t_critical_95`, `df = 4 → 2.776`) is the wrong quantile
//! for it — at `df = 4` the one-sided value is `2.132` — and it is behind
//! `#[cfg(feature = "llm")]`, so it is invisible to CI's default-feature run.
//! It is deliberately not reused.
//!
//! **`n ≥ 5`, and interleaved.** §4.3: "`n = 3` sizes an effect and bounds no
//! variance: no σ-dependent status changes at `n < 5`." And the replicates must
//! alternate A,B,A,B,… — thermal state, JIT/graph-capture warm state and free
//! VRAM all drift across a sweep, and alternation is the only design that
//! cancels the drift. Both are refusals here, not warnings:
//! [`log_ratio_lcb`] returns `None` and the caller reports the point estimate
//! with `lcb95: null`.

use serde::{Deserialize, Serialize};

use super::join::{Ratio, RatioMethod};

/// §4.3 — the replicate floor. Below it there is no verdict.
pub const MIN_REPLICATES: usize = 5;

/// Which arm ran **first** in one replicate. Recorded per replicate so
/// interleaving is a property of the data rather than of a claim about it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArmOrder {
    /// Subject first, comparator second.
    SubjectFirst,
    /// Comparator first, subject second.
    ComparatorFirst,
}

/// One interleaved replicate: the same window statistic from both lanes, plus
/// the order they ran in.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplicatePair {
    /// The subject lane's value.
    pub subject: f64,
    /// The comparator lane's value.
    pub comparator: f64,
    /// Which lane ran first in this replicate.
    pub order: ArmOrder,
}

/// One-sided 95% Student-t critical value, `df` degrees of freedom.
///
/// The published table for `df = 1..=30`; `1.645` (the normal quantile) beyond,
/// which is where the t distribution has converged to within 0.01. `df = 0` has
/// no bound and returns infinity, so a caller that reaches it produces
/// `lcb95 = −∞` rather than a plausible number.
#[must_use]
pub fn t_lower_one_sided_95(df: usize) -> f64 {
    const TABLE: [f64; 30] = [
        6.314, 2.920, 2.353, 2.132, 2.015, 1.943, 1.895, 1.860, 1.833, 1.812, 1.796, 1.782, 1.771,
        1.761, 1.753, 1.746, 1.740, 1.734, 1.729, 1.725, 1.721, 1.717, 1.714, 1.711, 1.708, 1.706,
        1.703, 1.701, 1.699, 1.697,
    ];
    match df {
        0 => f64::INFINITY,
        d if d <= TABLE.len() => TABLE[d - 1],
        _ => 1.645,
    }
}

/// §4.3 — the exponentiated one-sided 95% lower bound on the mean log-ratio.
///
/// Returns `None` — no verdict — when
///
/// - fewer than [`MIN_REPLICATES`] pairs were supplied, or
/// - the pairs are not strictly alternating (`order` must flip every replicate),
/// - or any pair carries a non-positive value, where `ln` is undefined.
///
/// A caller that gets `None` reports [`log_ratio_point`] instead, which carries
/// `lcb95: null` and the same `n`, so the ratio is still on the receipt and
/// still cannot be used as a verdict.
#[must_use]
pub fn log_ratio_lcb(pairs: &[ReplicatePair]) -> Option<Ratio> {
    if pairs.len() < MIN_REPLICATES || !is_strictly_alternating(pairs) {
        return None;
    }
    let logs = log_ratios(pairs)?;
    let n = logs.len();
    let mean = logs.iter().sum::<f64>() / n as f64;
    let var = logs.iter().map(|l| (l - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    let se = (var / n as f64).sqrt();
    let t = t_lower_one_sided_95(n - 1);
    Some(Ratio {
        point: mean.exp(),
        lcb95: Some(t.mul_add(-se, mean).exp()),
        method: RatioMethod::ReplicateTLower,
        n,
    })
}

/// The point estimate alone — `exp(mean(ln(subject/comparator)))` — for a
/// design that cannot support a bound.
///
/// `lcb95` is `None`, never the point estimate and never `0.0`: §4.3's "`n < 5`
/// → reporting only" has to be visible in the receipt, and a bound equal to the
/// point estimate would read as impossible precision.
#[must_use]
pub fn log_ratio_point(pairs: &[ReplicatePair]) -> Option<Ratio> {
    let logs = log_ratios(pairs)?;
    let n = logs.len();
    let mean = logs.iter().sum::<f64>() / n as f64;
    Some(Ratio::reporting_only(
        mean.exp(),
        RatioMethod::ReplicateTLower,
        n,
    ))
}

/// [`log_ratio_lcb`] when the design supports it, [`log_ratio_point`] otherwise.
/// The receipt always carries a ratio; only the bound is conditional.
#[must_use]
pub fn log_ratio_bound_or_point(pairs: &[ReplicatePair]) -> Option<Ratio> {
    log_ratio_lcb(pairs).or_else(|| log_ratio_point(pairs))
}

fn log_ratios(pairs: &[ReplicatePair]) -> Option<Vec<f64>> {
    if pairs.len() < 2 {
        return None;
    }
    pairs
        .iter()
        .map(|p| {
            if p.subject > 0.0 && p.comparator > 0.0 {
                Some((p.subject / p.comparator).ln())
            } else {
                None
            }
        })
        .collect()
}

/// §4.3 — A,B,A,B,…: the arm that goes first must flip every replicate.
fn is_strictly_alternating(pairs: &[ReplicatePair]) -> bool {
    pairs.windows(2).all(|w| w[0].order != w[1].order)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn alternating(values: &[(f64, f64)]) -> Vec<ReplicatePair> {
        values
            .iter()
            .enumerate()
            .map(|(i, &(subject, comparator))| ReplicatePair {
                subject,
                comparator,
                order: if i % 2 == 0 {
                    ArmOrder::SubjectFirst
                } else {
                    ArmOrder::ComparatorFirst
                },
            })
            .collect()
    }

    /// The published one-sided 95% t values. If this reds, every bound moved.
    #[test]
    fn one_sided_t_table_matches_published_values() {
        for (df, want) in [
            (1_usize, 6.314_f64),
            (2, 2.920),
            (3, 2.353),
            (4, 2.132),
            (5, 2.015),
            (6, 1.943),
            (7, 1.895),
            (8, 1.860),
            (9, 1.833),
            (10, 1.812),
            (11, 1.796),
            (12, 1.782),
            (13, 1.771),
            (14, 1.761),
            (15, 1.753),
            (16, 1.746),
            (17, 1.740),
            (18, 1.734),
            (19, 1.729),
            (20, 1.725),
            (21, 1.721),
            (22, 1.717),
            (23, 1.714),
            (24, 1.711),
            (25, 1.708),
            (26, 1.706),
            (27, 1.703),
            (28, 1.701),
            (29, 1.699),
            (30, 1.697),
        ] {
            assert_eq!(t_lower_one_sided_95(df), want, "df={df}");
        }
        assert_eq!(t_lower_one_sided_95(31), 1.645, "beyond 30, the normal");
        assert_eq!(t_lower_one_sided_95(1_000), 1.645);
        assert!(
            t_lower_one_sided_95(0).is_infinite(),
            "df=0 supports no bound"
        );
    }

    /// It is the ONE-SIDED table. The two-tailed 95% value at df=4 is 2.776;
    /// using it would widen every bound by 30% and silently pass regressions.
    #[test]
    fn the_table_is_one_sided_not_two_tailed() {
        assert_eq!(t_lower_one_sided_95(4), 2.132);
        assert_ne!(t_lower_one_sided_95(4), 2.776);
    }

    /// §4.3 — `n = 3` bounds no variance, so there is no bound.
    #[test]
    fn fewer_than_five_replicates_give_no_bound() {
        let three = alternating(&[(100.0, 90.0), (101.0, 91.0), (99.0, 89.0)]);
        assert!(log_ratio_lcb(&three).is_none());
        assert_eq!(MIN_REPLICATES, 5);

        // …but the point estimate is still reported, with a null bound.
        let reporting = log_ratio_point(&three).expect("point estimate exists");
        assert!(reporting.lcb95.is_none());
        assert_eq!(reporting.n, 3);
        assert!(!reporting.passes(0.0), "no bound is not a pass");

        // And five is enough.
        let five = alternating(&[
            (100.0, 90.0),
            (101.0, 91.0),
            (99.0, 89.0),
            (100.5, 90.5),
            (100.2, 90.1),
        ]);
        assert!(log_ratio_lcb(&five).is_some());
    }

    /// §4.3 — replicates that did not alternate did not cancel the drift, so
    /// they do not carry a bound however many of them there are.
    #[test]
    fn non_alternating_order_is_refused() {
        let mut pairs = alternating(&[
            (100.0, 90.0),
            (101.0, 91.0),
            (99.0, 89.0),
            (100.5, 90.5),
            (100.2, 90.1),
        ]);
        assert!(log_ratio_lcb(&pairs).is_some(), "control: alternating");
        pairs[3].order = pairs[2].order;
        assert!(
            log_ratio_lcb(&pairs).is_none(),
            "two consecutive replicates led with the same arm"
        );
        // The point estimate survives; only the verdict is withdrawn.
        assert!(log_ratio_point(&pairs).is_some());
    }

    /// The bound is a bound on the RATIO: the estimate is formed in log space
    /// and exponentiated, so `point` is the geometric mean of the per-replicate
    /// ratios and `lcb95` sits below it.
    #[test]
    fn log_ratio_bound_is_exponentiated() {
        // Every replicate is exactly 1.10, so the geometric mean is 1.10 and the
        // spread is zero: the bound coincides with the point.
        let flat = alternating(&[
            (110.0, 100.0),
            (220.0, 200.0),
            (55.0, 50.0),
            (11.0, 10.0),
            (1100.0, 1000.0),
        ]);
        let r = log_ratio_lcb(&flat).expect("n = 5, alternating");
        assert!((r.point - 1.10).abs() < 1e-12, "{r:?}");
        assert!(
            (r.lcb95.expect("bounded") - 1.10).abs() < 1e-12,
            "zero variance leaves the bound at the point: {r:?}"
        );
        assert_eq!(r.method, RatioMethod::ReplicateTLower);
        assert_eq!(r.n, 5);

        // An arithmetic mean of the ratios would give a different centre for a
        // skewed set; the geometric mean of 0.5 and 2.0 is 1.0, not 1.25.
        let skewed = alternating(&[
            (50.0, 100.0),
            (200.0, 100.0),
            (50.0, 100.0),
            (200.0, 100.0),
            (100.0, 100.0),
        ]);
        let g = log_ratio_lcb(&skewed).expect("n = 5");
        assert!((g.point - 1.0).abs() < 1e-12, "geometric mean: {g:?}");
        assert!(g.lcb95.expect("bounded") < g.point, "{g:?}");
    }

    /// The bound must MOVE with dispersion, or `t · se` is decoration.
    #[test]
    fn more_dispersion_lowers_the_bound() {
        let tight = alternating(&[
            (110.0, 100.0),
            (109.0, 100.0),
            (111.0, 100.0),
            (110.5, 100.0),
            (109.5, 100.0),
        ]);
        let loose = alternating(&[
            (60.0, 100.0),
            (160.0, 100.0),
            (70.0, 100.0),
            (150.0, 100.0),
            (110.0, 100.0),
        ]);
        let a = log_ratio_lcb(&tight).expect("n = 5");
        let b = log_ratio_lcb(&loose).expect("n = 5");
        assert!(
            b.lcb95.expect("bounded") < a.lcb95.expect("bounded"),
            "dispersed {b:?} must bound lower than tight {a:?}"
        );
    }

    /// One replicate is not a paired design: there is nothing to average and
    /// nothing to bound, so there is no ratio at all rather than a ratio of
    /// impossible precision.
    #[test]
    fn a_single_replicate_has_no_log_ratio() {
        let one = alternating(&[(110.0, 100.0)]);
        assert!(log_ratio_point(&one).is_none());
        assert!(log_ratio_lcb(&one).is_none());
        assert!(log_ratio_bound_or_point(&one).is_none());
        assert!(log_ratio_point(&[]).is_none());
        // Two is enough for a point estimate, still not for a bound.
        let two = alternating(&[(110.0, 100.0), (90.0, 100.0)]);
        let r = log_ratio_point(&two).expect("two pairs give a point");
        assert_eq!(r.n, 2);
        assert!(r.lcb95.is_none());
    }

    /// A non-positive lane value has no logarithm, and a zero-throughput lane is
    /// not a ratio of any kind.
    #[test]
    fn a_zero_lane_has_no_log_ratio() {
        let zeroed = alternating(&[
            (110.0, 100.0),
            (0.0, 100.0),
            (111.0, 100.0),
            (110.5, 100.0),
            (109.5, 100.0),
        ]);
        assert!(log_ratio_lcb(&zeroed).is_none());
        assert!(log_ratio_point(&zeroed).is_none());
    }

    /// The convenience wrapper reports when it cannot bound, and bounds when it
    /// can — so a caller never has to choose between "no ratio" and "a ratio
    /// that pretends to a verdict".
    #[test]
    fn the_wrapper_falls_back_to_reporting_only() {
        let three = alternating(&[(100.0, 90.0), (101.0, 91.0), (99.0, 89.0)]);
        let r = log_ratio_bound_or_point(&three).expect("point estimate");
        assert!(r.lcb95.is_none());
        let five = alternating(&[
            (100.0, 90.0),
            (101.0, 91.0),
            (99.0, 89.0),
            (100.5, 90.5),
            (100.2, 90.1),
        ]);
        assert!(log_ratio_bound_or_point(&five)
            .expect("bounded")
            .lcb95
            .is_some());
    }
}
