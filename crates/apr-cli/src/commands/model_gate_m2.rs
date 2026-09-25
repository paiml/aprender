//! M2 quality gate statistics (EXT-001 §3.6, row EXT-12, aprender#4394).
//!
//! §3.6, verbatim: "Sealed suites, paired candidate-vs-incumbent: McNemar exact for
//! binary items, seeded bootstrap CI otherwise, Holm across suites. α is
//! pre-registered in the contract (basis: REX-001 §3). **Promote iff** no suite is
//! significantly worse **and** (≥1 suite is significantly better **or** the release
//! class is PATCH/packaging with tensor-identical files **or** it is the first release,
//! which reports absolute levels and claims no improvement). Underpowered by the
//! sample-size rule → no promotion (S-10)."
//!
//! Every number this module decides with comes from [`M2Prereg`]; none is chosen here
//! (R-13). The definitions replicate REX-001's pre-registered analysis plan
//! (`docs/audits/rex-001/analysis-plan.md`, crate `aprender-review-experiment`
//! `stats.rs`) so the two experiments compute identical numbers. That crate is not on
//! `main` yet; when it lands, this module should call it instead of duplicating it.
//! One deliberate difference: a bootstrap p counts resamples `≤ 0` (not `< 0`), so two
//! identical arms are neither better nor worse. REX's `< 0` would call them better.
//!
//! The baseline is any paired arm — the incumbent for M2, upstream stock for EXT-30.

// Consumed by `apr model gate`, the rest of EXT-12; until then only tests call it.
#![cfg_attr(not(test), allow(dead_code))]

use serde::Serialize;
use std::fmt;

/// The pre-registered constants M2 decides with.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub(crate) struct M2Prereg {
    /// Family-wise α for each Holm family (worse, better).
    pub alpha: f64,
    /// Normal quantile for the Wilson interval.
    pub z: f64,
    /// Sample-size rule for binary suites: the candidate's Wilson half-width must not exceed this.
    pub max_wilson_half_width: f64,
    /// SplitMix64 seed for every bootstrap.
    pub bootstrap_seed: u64,
    /// Resamples per bootstrap.
    pub bootstrap_resamples: usize,
}

impl M2Prereg {
    /// REX-001 §3: α 0.05 with Holm; Wilson z for 95 %; half-width 0.12 (§2.2);
    /// 10 000 paired resamples from SplitMix64 seed 4354.
    pub const REX_001: Self = Self {
        alpha: 0.05,
        z: 1.959_963_984_540_054,
        max_wilson_half_width: 0.12,
        bootstrap_seed: 4354,
        bootstrap_resamples: 10_000,
    };
}

/// One sealed suite's per-item results. Items are paired by index.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum SuiteData {
    /// Pass/fail per item: McNemar exact.
    Binary {
        candidate: Vec<bool>,
        /// `None` only for a first release.
        baseline: Option<Vec<bool>>,
    },
    /// A score per item: paired bootstrap of the mean difference.
    Scored {
        candidate: Vec<f64>,
        baseline: Option<Vec<f64>>,
        higher_is_better: bool,
        /// The suite's own pre-registered sample-size rule, in its units: the 95 % bootstrap
        /// CI half-width must not exceed this.
        max_ci_half_width: f64,
    },
}

/// A sealed suite.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Suite {
    pub name: String,
    pub data: SuiteData,
}

/// §3.6 release classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ReleaseClass {
    /// Must be significantly better on at least one suite.
    Improvement,
    /// PATCH/packaging with tensor-identical files: must only be no worse.
    PatchTensorIdentical,
    /// No incumbent: reports absolute levels, claims no improvement.
    First,
}

/// Holm result for one hypothesis.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub(crate) struct Holm {
    pub adjusted_p: f64,
    pub rejected: bool,
}

/// What M2 found on one suite.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct SuiteReport {
    pub name: String,
    pub n: usize,
    /// Candidate pass rate (binary) or mean score (scored): the absolute level.
    pub candidate_level: f64,
    /// The quantity the sample-size rule reads, and its pre-registered maximum.
    pub half_width: f64,
    pub max_half_width: f64,
    /// Items needed at this point estimate, when the suite is underpowered.
    pub n_req: Option<u64>,
    /// Discordant pairs (binary): candidate-only passes, baseline-only passes.
    pub discordant: Option<(u64, u64)>,
    /// 95 % percentile CI of the oriented mean difference (scored).
    pub diff_ci: Option<(f64, f64)>,
    pub p_better: Option<f64>,
    pub p_worse: Option<f64>,
    pub holm_better: Option<Holm>,
    pub holm_worse: Option<Holm>,
}

/// The M2 verdict.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub(crate) enum Verdict {
    Promote,
    Reject {
        reason: String,
    },
    /// S-10: the largest required n across underpowered suites, and that suite.
    Underpowered {
        n_req: u64,
        suite: String,
    },
}

impl fmt::Display for Verdict {
    /// The `verdict=` field of the §9 `quality:` status line.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Promote => write!(f, "promote"),
            Self::Reject { .. } => write!(f, "reject"),
            Self::Underpowered { n_req, .. } => write!(f, "underpowered({n_req})"),
        }
    }
}

/// The M2 section of `model-gate-receipt-v1`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct M2Report {
    pub prereg: M2Prereg,
    pub class: ReleaseClass,
    pub suites: Vec<SuiteReport>,
    pub verdict: Verdict,
}

// ---- REX-001 statistics -------------------------------------------------------------

/// Wilson score interval for `k` of `n`, clamped to [0, 1]. `None` when `n == 0`.
pub(crate) fn wilson(k: u64, n: u64, z: f64) -> Option<(f64, f64)> {
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

/// Half-width of the Wilson interval.
pub(crate) fn wilson_half_width(k: u64, n: u64, z: f64) -> Option<f64> {
    wilson(k, n, z).map(|(lo, hi)| (hi - lo) / 2.0)
}

/// Smallest `m > n` whose Wilson half-width at the point estimate `k/n` meets `max`.
fn wilson_n_req(k: u64, n: u64, z: f64, max: f64) -> u64 {
    let p = k as f64 / n as f64;
    // At p = ½ (the widest case) a 0.12 rule needs 63; the cap only guards a max near 0.
    (n + 1..=10_000_000)
        .find(|&m| {
            let km = ((p * m as f64).round() as u64).min(m);
            wilson_half_width(km, m, z).is_some_and(|h| h <= max)
        })
        .unwrap_or(10_000_000)
}

fn ln_choose(n: u64, k: u64) -> f64 {
    let k = k.min(n - k);
    (0..k)
        .map(|i| ((n - i) as f64).ln() - ((i + 1) as f64).ln())
        .sum()
}

/// McNemar exact, one-sided: P(X ≥ b), X ~ Bin(b + c, ½). No discordant pairs: p = 1.
pub(crate) fn mcnemar_exact_one_sided(b: u64, c: u64) -> f64 {
    let n = b + c;
    if n == 0 {
        return 1.0;
    }
    let ln_half_n = (n as f64) * 0.5_f64.ln();
    let p: f64 = (b..=n).map(|x| (ln_choose(n, x) + ln_half_n).exp()).sum();
    p.min(1.0)
}

/// SplitMix64, the pre-registered bootstrap generator.
#[derive(Debug, Clone)]
pub(crate) struct SplitMix64(u64);

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform index in `0..n` (Lemire multiply-shift; `n > 0`).
    pub(crate) fn below(&mut self, n: usize) -> usize {
        ((u128::from(self.next_u64()) * n as u128) >> 64) as usize
    }
}

/// Nearest-rank percentile of sorted values: the `ceil(q·n)`-th, at least the first.
pub(crate) fn percentile_sorted(sorted: &[f64], q: f64) -> f64 {
    let rank = ((q * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len());
    sorted[rank - 1]
}

/// Bootstrap of the mean of `d`: 95 % percentile CI, and the shares of resample means
/// `≤ 0` (the p for "mean > 0") and `≥ 0` (the p for "mean < 0").
pub(crate) struct Bootstrap {
    pub ci: (f64, f64),
    pub share_le_zero: f64,
    pub share_ge_zero: f64,
}

pub(crate) fn bootstrap_mean(d: &[f64], seed: u64, resamples: usize) -> Bootstrap {
    let mut rng = SplitMix64::new(seed);
    let n = d.len();
    let mut means: Vec<f64> = (0..resamples)
        .map(|_| (0..n).map(|_| d[rng.below(n)]).sum::<f64>() / n as f64)
        .collect();
    means.sort_by(f64::total_cmp);
    let share =
        |f: fn(f64) -> bool| means.iter().filter(|&&m| f(m)).count() as f64 / resamples as f64;
    Bootstrap {
        ci: (
            percentile_sorted(&means, 0.025),
            percentile_sorted(&means, 0.975),
        ),
        share_le_zero: share(|m| m <= 0.0),
        share_ge_zero: share(|m| m >= 0.0),
    }
}

/// Holm step-down at family-wise `alpha`, in input order.
pub(crate) fn holm(p: &[f64], alpha: f64) -> Vec<Holm> {
    let m = p.len();
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| p[a].total_cmp(&p[b]).then(a.cmp(&b)));
    let mut out = vec![
        Holm {
            adjusted_p: 1.0,
            rejected: false
        };
        m
    ];
    let (mut running, mut still_rejecting) = (0.0_f64, true);
    for (rank, &i) in order.iter().enumerate() {
        let factor = (m - rank) as f64;
        running = running.max((p[i] * factor).min(1.0));
        still_rejecting = still_rejecting && p[i] <= alpha / factor;
        out[i] = Holm {
            adjusted_p: running,
            rejected: still_rejecting,
        };
    }
    out
}

// ---- the gate -----------------------------------------------------------------------

fn validate(class: ReleaseClass, suites: &[Suite]) -> Result<(), String> {
    if suites.is_empty() {
        return Err("M2: no sealed suites".into());
    }
    let mut names = std::collections::HashSet::new();
    for s in suites {
        if !names.insert(s.name.as_str()) {
            return Err(format!("M2: suite {} appears twice", s.name));
        }
        let (n, baseline_len, finite, rule_ok) = match &s.data {
            SuiteData::Binary {
                candidate,
                baseline,
            } => (candidate.len(), baseline.as_ref().map(Vec::len), true, true),
            SuiteData::Scored {
                candidate,
                baseline,
                max_ci_half_width,
                ..
            } => (
                candidate.len(),
                baseline.as_ref().map(Vec::len),
                candidate
                    .iter()
                    .chain(baseline.iter().flatten())
                    .all(|x| x.is_finite()),
                max_ci_half_width.is_finite() && *max_ci_half_width > 0.0,
            ),
        };
        let bad = |why: &str| Err(format!("M2: suite {}: {why}", s.name));
        if n == 0 {
            return bad("no items");
        }
        if !finite {
            return bad("a score is not finite");
        }
        if !rule_ok {
            return bad("no pre-registered max CI half-width");
        }
        match (class, baseline_len) {
            (ReleaseClass::First, Some(_)) => return bad("a first release has no incumbent"),
            (ReleaseClass::First, None) => {}
            (_, None) => return bad("no baseline arm to compare against"),
            (_, Some(b)) if b != n => return bad(&format!("{n} candidate items, {b} baseline")),
            (_, Some(_)) => {}
        }
    }
    Ok(())
}

fn binary_report(pre: &M2Prereg, name: &str, cand: &[bool], base: Option<&[bool]>) -> SuiteReport {
    let n = cand.len() as u64;
    let k = cand.iter().filter(|&&x| x).count() as u64;
    let half_width = wilson_half_width(k, n, pre.z).expect("validated: n > 0");
    let powered = half_width <= pre.max_wilson_half_width;
    let discordant = base.map(|base| {
        let b = cand.iter().zip(base).filter(|(c, x)| **c && !**x).count() as u64;
        let c = cand.iter().zip(base).filter(|(c, x)| !**c && **x).count() as u64;
        (b, c)
    });
    SuiteReport {
        name: name.to_string(),
        n: cand.len(),
        candidate_level: k as f64 / n as f64,
        half_width,
        max_half_width: pre.max_wilson_half_width,
        n_req: (!powered).then(|| wilson_n_req(k, n, pre.z, pre.max_wilson_half_width)),
        discordant,
        diff_ci: None,
        p_better: discordant.map(|(b, c)| mcnemar_exact_one_sided(b, c)),
        p_worse: discordant.map(|(b, c)| mcnemar_exact_one_sided(c, b)),
        holm_better: None,
        holm_worse: None,
    }
}

fn scored_report(
    pre: &M2Prereg,
    name: &str,
    cand: &[f64],
    base: Option<&[f64]>,
    higher_is_better: bool,
    max: f64,
) -> SuiteReport {
    let n = cand.len();
    let sign = if higher_is_better { 1.0 } else { -1.0 };
    // Oriented so that > 0 is better; with no baseline, the candidate's own level.
    let d: Vec<f64> = match base {
        Some(base) => cand.iter().zip(base).map(|(c, b)| sign * (c - b)).collect(),
        None => cand.to_vec(),
    };
    let boot = bootstrap_mean(&d, pre.bootstrap_seed, pre.bootstrap_resamples);
    let half_width = (boot.ci.1 - boot.ci.0) / 2.0;
    // CI width scales as 1/√n, so n·(w/max)² items reach the rule at this spread.
    let n_req = (half_width > max).then(|| (n as f64 * (half_width / max).powi(2)).ceil() as u64);
    SuiteReport {
        name: name.to_string(),
        n,
        candidate_level: cand.iter().sum::<f64>() / n as f64,
        half_width,
        max_half_width: max,
        n_req,
        discordant: None,
        diff_ci: base.map(|_| boot.ci),
        p_better: base.map(|_| boot.share_le_zero),
        p_worse: base.map(|_| boot.share_ge_zero),
        holm_better: None,
        holm_worse: None,
    }
}

fn suite_report(pre: &M2Prereg, s: &Suite) -> SuiteReport {
    match &s.data {
        SuiteData::Binary {
            candidate,
            baseline,
        } => binary_report(pre, &s.name, candidate, baseline.as_deref()),
        SuiteData::Scored {
            candidate,
            baseline,
            higher_is_better,
            max_ci_half_width,
        } => scored_report(
            pre,
            &s.name,
            candidate,
            baseline.as_deref(),
            *higher_is_better,
            *max_ci_half_width,
        ),
    }
}

fn apply_holm(pre: &M2Prereg, reports: &mut [SuiteReport]) {
    let better: Vec<f64> = reports.iter().filter_map(|r| r.p_better).collect();
    let worse: Vec<f64> = reports.iter().filter_map(|r| r.p_worse).collect();
    if better.len() != reports.len() {
        return; // a first release runs no comparison
    }
    for ((r, hb), hw) in reports
        .iter_mut()
        .zip(holm(&better, pre.alpha))
        .zip(holm(&worse, pre.alpha))
    {
        r.holm_better = Some(hb);
        r.holm_worse = Some(hw);
    }
}

fn decide(class: ReleaseClass, reports: &[SuiteReport]) -> Verdict {
    if let Some(r) = reports
        .iter()
        .find(|r| r.holm_worse.is_some_and(|h| h.rejected))
    {
        return Verdict::Reject {
            reason: format!("suite {} is significantly worse", r.name),
        };
    }
    if let Some(r) = reports
        .iter()
        .filter(|r| r.n_req.is_some())
        .max_by_key(|r| r.n_req)
    {
        return Verdict::Underpowered {
            n_req: r.n_req.unwrap_or_default(),
            suite: r.name.clone(),
        };
    }
    let better = reports
        .iter()
        .any(|r| r.holm_better.is_some_and(|h| h.rejected));
    match class {
        ReleaseClass::First | ReleaseClass::PatchTensorIdentical => Verdict::Promote,
        ReleaseClass::Improvement if better => Verdict::Promote,
        ReleaseClass::Improvement => Verdict::Reject {
            reason: "no suite is significantly better".into(),
        },
    }
}

/// Run M2 over the sealed suites.
///
/// # Errors
///
/// Refuses (no verdict) when the suites cannot be compared: none, duplicate names, empty
/// or mismatched arms, a missing baseline outside a first release, a baseline on a first
/// release, a non-finite score, or a scored suite with no pre-registered rule.
pub(crate) fn m2(
    pre: &M2Prereg,
    class: ReleaseClass,
    suites: &[Suite],
) -> Result<M2Report, String> {
    validate(class, suites)?;
    let mut reports: Vec<SuiteReport> = suites.iter().map(|s| suite_report(pre, s)).collect();
    apply_holm(pre, &mut reports);
    let verdict = decide(class, &reports);
    Ok(M2Report {
        prereg: *pre,
        class,
        suites: reports,
        verdict,
    })
}

#[cfg(test)]
#[path = "model_gate_m2_tests.rs"]
mod tests;
