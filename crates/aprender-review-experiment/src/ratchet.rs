//! REX-10 perf ratchet (§5.1): review p95 on the primary cell may not rise.
//!
//! Every tag reruns the perf arm on the primary cell over one fixed item set
//! (full dev split + 30 fixed test items, timing only). `record` turns that
//! tag's warm, non-rerun receipts into one `review-lane-perf-ratchet-v1` entry
//! holding each item's wall time. `check` reads the entries of one cell in
//! record order:
//! - fewer than [`ARM_AFTER`] tags: **arming**, nothing to compare yet;
//! - each later tag is paired by item with the best (lowest-p95) earlier tag,
//!   and a paired bootstrap gives a CI for `p95(new) / p95(best)`, each p95
//!   the Harrell–Davis estimate. A CI whose lower bound is above 1 is a
//!   regression and the andon is RED.
//!
//! The paired CI, not "the two p95 CIs overlap", is the rule: the same items
//! are timed every tag, so pairing removes item-to-item spread. The unpaired
//! overlap rule could not see a +10 % p95 regression at the pilot's n
//! (FALSIFY-RXR-002 measures this). Nor could a paired nearest-rank p95: it is
//! one order statistic, so one item's run-to-run noise decides it; the
//! Harrell–Davis estimate of the same quantile weights the whole tail.
//! The only level used is the prereg's [`ALPHA`]; no noise threshold is
//! invented (R-7).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::stats::{percentile, percentile_sorted, SplitMix64, ALPHA, BOOTSTRAP_RESAMPLES};

pub const SCHEME: &str = "review-lane-perf-ratchet-v1";
/// The ratchet holds after this many recorded tags (§5.1).
pub const ARM_AFTER: usize = 3;
const P95: f64 = 0.95;
/// The harness seed (`harness::DECODING.seed`), so every CI reproduces.
const SEED: u64 = 4354;

/// One recorded tag on one cell.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    pub schema: String,
    pub tag: String,
    pub cell: String,
    pub apr_sha256: String,
    /// Warm wall time per item, ms (the median when an item ran twice).
    pub items: BTreeMap<String, f64>,
    pub p95_ms: f64,
    /// Unpaired percentile-bootstrap CI of this tag's p95, for display.
    pub p95_ci_ms: [f64; 2],
    /// The same p95 from llama.cpp on the same cell and items, if it ran.
    pub llama_cpp_p95_ms: Option<f64>,
    pub receipts_sha256: String,
}

/// A warm, non-rerun timing already filtered to one cell and one tag.
#[derive(Debug, Clone)]
pub struct Sample<'a> {
    pub item_id: &'a str,
    pub apr_sha256: &'a str,
    pub wall_ms: f64,
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// Per-item wall time: the median of each item's finite, positive timings.
fn per_item(samples: &[Sample<'_>]) -> Result<BTreeMap<String, f64>, String> {
    let mut by: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for s in samples {
        if !(s.wall_ms.is_finite() && s.wall_ms > 0.0) {
            return Err(format!(
                "{}: wall_ms {} is not a timing",
                s.item_id, s.wall_ms
            ));
        }
        by.entry(s.item_id.to_string()).or_default().push(s.wall_ms);
    }
    Ok(by
        .into_iter()
        .map(|(k, mut v)| (k, median(&mut v)))
        .collect())
}

fn ci_of(mut stat: Vec<f64>) -> [f64; 2] {
    stat.sort_by(f64::total_cmp);
    [
        percentile_sorted(&stat, ALPHA / 2.0),
        percentile_sorted(&stat, 1.0 - ALPHA / 2.0),
    ]
}

/// Unpaired percentile-bootstrap CI of the p95 of `x`.
fn p95_ci(x: &[f64]) -> [f64; 2] {
    let mut rng = SplitMix64::new(SEED);
    let mut buf = vec![0.0; x.len()];
    let stat = (0..BOOTSTRAP_RESAMPLES)
        .map(|_| {
            for b in &mut buf {
                *b = x[rng.below(x.len())];
            }
            percentile(&buf, P95)
        })
        .collect();
    ci_of(stat)
}

/// Build one tag's entry. A tag with no timings did not run and is not a
/// recorded tag (R-6); mixed apr binaries under one tag are refused.
pub fn record(
    tag: &str,
    cell: &str,
    samples: &[Sample<'_>],
    llama_cpp: &[Sample<'_>],
    receipts_sha256: &str,
) -> Result<Entry, String> {
    let shas: std::collections::BTreeSet<&str> = samples.iter().map(|s| s.apr_sha256).collect();
    let apr_sha256 = match shas.len() {
        0 => {
            return Err(format!(
                "{tag} on {cell}: no warm timings; the tag did not run"
            ))
        }
        1 => shas.into_iter().next().unwrap_or_default().to_string(),
        n => {
            return Err(format!(
                "{tag} on {cell}: {n} different apr binaries under one tag"
            ))
        }
    };
    let items = per_item(samples)?;
    let v: Vec<f64> = items.values().copied().collect();
    let llama_cpp_p95_ms = if llama_cpp.is_empty() {
        None
    } else {
        let l: Vec<f64> = per_item(llama_cpp)?.into_values().collect();
        Some(percentile(&l, P95))
    };
    Ok(Entry {
        schema: SCHEME.into(),
        tag: tag.into(),
        cell: cell.into(),
        apr_sha256,
        p95_ms: percentile(&v, P95),
        p95_ci_ms: p95_ci(&v),
        items,
        llama_cpp_p95_ms,
        receipts_sha256: receipts_sha256.into(),
    })
}

/// ln Γ(x), Lanczos (g = 7, n = 9); x > 0.
fn ln_gamma(x: f64) -> f64 {
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let x = x - 1.0;
    let t = x + 7.5;
    let s = C[1..]
        .iter()
        .enumerate()
        .fold(C[0], |s, (i, c)| s + c / (x + i as f64 + 1.0));
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + s.ln()
}

fn not_tiny(v: f64) -> f64 {
    if v.abs() < 1e-300 {
        1e-300
    } else {
        v
    }
}

/// Continued fraction of the incomplete beta (modified Lentz).
fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let mut c = 1.0;
    let mut d = not_tiny(1.0 - qab * x / qap).recip();
    let mut h = d;
    for m in 1..=300 {
        let m = f64::from(m);
        let m2 = 2.0 * m;
        for aa in [
            m * (b - m) * x / ((qam + m2) * (a + m2)),
            -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2)),
        ] {
            d = not_tiny(1.0 + aa * d).recip();
            c = not_tiny(1.0 + aa / c);
            h *= d * c;
        }
        if (d * c - 1.0).abs() < 1e-15 {
            break;
        }
    }
    h
}

/// Regularized incomplete beta I_x(a, b).
fn inc_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let front =
        (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        front * beta_cf(a, b, x) / a
    } else {
        1.0 - front * beta_cf(b, a, 1.0 - x) / b
    }
}

/// Harrell–Davis weights for quantile `q` of `n` sorted values: a weighted
/// sum of every order statistic, so one noisy tail item cannot swing it the
/// way it swings the nearest-rank p95.
#[must_use]
pub fn harrell_davis_weights(n: usize, q: f64) -> Vec<f64> {
    let (a, b) = ((n as f64 + 1.0) * q, (n as f64 + 1.0) * (1.0 - q));
    (1..=n)
        .map(|i| inc_beta(a, b, i as f64 / n as f64) - inc_beta(a, b, (i - 1) as f64 / n as f64))
        .collect()
}

fn hd(buf: &mut [f64], w: &[f64]) -> f64 {
    buf.sort_by(f64::total_cmp);
    buf.iter().zip(w).map(|(x, w)| x * w).sum()
}

/// Harrell–Davis p95 ratio `new / reference` over paired items, and its
/// paired-bootstrap CI (items resampled together, so both sides see the
/// same items).
#[must_use]
pub fn paired_p95_ratio_ci(pairs: &[(f64, f64)]) -> Option<(f64, [f64; 2])> {
    if pairs.is_empty() {
        return None;
    }
    let w = harrell_davis_weights(pairs.len(), P95);
    let (mut a, mut b): (Vec<f64>, Vec<f64>) = pairs.iter().copied().unzip();
    let point = hd(&mut a, &w) / hd(&mut b, &w);
    let mut rng = SplitMix64::new(SEED);
    let stat = (0..BOOTSTRAP_RESAMPLES)
        .map(|_| {
            for i in 0..pairs.len() {
                let (n, r) = pairs[rng.below(pairs.len())];
                a[i] = n;
                b[i] = r;
            }
            hd(&mut a, &w) / hd(&mut b, &w)
        })
        .collect();
    Some((point, ci_of(stat)))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Andon {
    Arming,
    Green,
    Red,
}

/// One armed comparison: `tag` against the best earlier `reference`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Comparison {
    pub tag: String,
    pub reference: String,
    pub paired_items: usize,
    pub p95_ratio: f64,
    pub p95_ratio_ci: [f64; 2],
    pub regression: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Verdict {
    pub andon: Andon,
    pub cell: Option<String>,
    pub tags: usize,
    pub comparisons: Vec<Comparison>,
    pub violations: Vec<String>,
    pub notes: Vec<String>,
}

/// Read the ratchet file of one cell and decide the andon. Violations (a
/// foreign schema or cell, a duplicate tag, no common items) are RED too:
/// the ratchet fails closed.
#[must_use]
pub fn check(entries: &[Entry]) -> Verdict {
    let mut v = Verdict {
        andon: Andon::Arming,
        cell: entries.first().map(|e| e.cell.clone()),
        tags: entries.len(),
        comparisons: Vec::new(),
        violations: Vec::new(),
        notes: Vec::new(),
    };
    let mut seen = std::collections::BTreeSet::new();
    for e in entries {
        if e.schema != SCHEME {
            v.violations
                .push(format!("{}: schema {} is not {SCHEME}", e.tag, e.schema));
        }
        if Some(&e.cell) != v.cell.as_ref() {
            v.violations
                .push(format!("{}: cell {} is not the file's cell", e.tag, e.cell));
        }
        if !seen.insert(e.tag.as_str()) {
            v.violations.push(format!("{}: tag recorded twice", e.tag));
        }
    }
    for (i, e) in entries.iter().enumerate().skip(ARM_AFTER) {
        let best = entries[..i]
            .iter()
            .min_by(|a, b| a.p95_ms.total_cmp(&b.p95_ms))
            .unwrap_or(&entries[0]);
        let pairs: Vec<(f64, f64)> = e
            .items
            .iter()
            .filter_map(|(k, n)| best.items.get(k).map(|r| (*n, *r)))
            .collect();
        if pairs.len() != e.items.len() || pairs.len() != best.items.len() {
            v.notes.push(format!(
                "{} vs {}: {} common items of {} / {}",
                e.tag,
                best.tag,
                pairs.len(),
                e.items.len(),
                best.items.len()
            ));
        }
        let Some((ratio, ci)) = paired_p95_ratio_ci(&pairs) else {
            v.violations
                .push(format!("{} vs {}: no common items", e.tag, best.tag));
            continue;
        };
        v.comparisons.push(Comparison {
            tag: e.tag.clone(),
            reference: best.tag.clone(),
            paired_items: pairs.len(),
            p95_ratio: ratio,
            p95_ratio_ci: ci,
            regression: ci[0] > 1.0,
        });
    }
    for w in entries.windows(2) {
        let gap = |e: &Entry| e.llama_cpp_p95_ms.map(|l| e.p95_ms / l);
        if let (Some(g0), Some(g1)) = (gap(&w[0]), gap(&w[1])) {
            if g1 > g0 {
                v.notes.push(format!(
                    "{}: gap to llama.cpp grew {g0:.3}× → {g1:.3}× (§9.1: file it)",
                    w[1].tag
                ));
            }
        }
    }
    v.andon = if !v.violations.is_empty() || v.comparisons.iter().any(|c| c.regression) {
        Andon::Red
    } else if entries.len() > ARM_AFTER {
        Andon::Green
    } else {
        Andon::Arming
    };
    v
}

#[cfg(test)]
#[path = "ratchet_tests.rs"]
mod tests;
