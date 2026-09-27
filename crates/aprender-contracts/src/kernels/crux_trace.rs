//! CRUX tracing metrics (TRACE-001 §2.4, row TR-11).
//!
//! Scalar reference implementations of the metrics bound by the four
//! tracing CRUX contracts:
//!
//! | Contract | Metric |
//! |---|---|
//! | `crux-F-22-v1` syscall-trace | E-S1 [`syscall_divergence`] + [`parity_ok`], E-S2 [`overhead_gap_effect`] |
//! | `crux-F-23-v1` golden-trace | §2.5 [`golden_theta`] + [`golden_verdict`] |
//! | `crux-F-24-v1` kernel-profile | E-S3 [`kernel_jaccard`] + [`kernel_times_ok`] |
//! | `crux-F-25-v1` op-attribution | [`op_attribution_verdict`] (Σ components ≤ wall, R-1) |
//!
//! Every gate threshold here comes from the reference tool's own
//! run-to-run noise (a quantile of self-divergence), never from an
//! invented constant. The integer cores ([`divergence_parts`],
//! [`jaccard_parts`]) carry the Kani obligations `KANI-CRUX-D` and
//! `KANI-CRUX-J`.

use std::collections::{BTreeMap, BTreeSet};

/// E-S1 integer core over two count vectors aligned by syscall.
///
/// Returns `(Σ_s |a_s − b_s|, Σ_s b_s)`; `D = num / den`. Vectors of
/// unequal length are aligned by treating missing entries as 0.
#[must_use]
pub fn divergence_parts(a: &[u64], b: &[u64]) -> (u64, u64) {
    let n = a.len().max(b.len());
    let mut num: u64 = 0;
    let mut den: u64 = 0;
    for i in 0..n {
        let x = a.get(i).copied().unwrap_or(0);
        let y = b.get(i).copied().unwrap_or(0);
        num = num.saturating_add(x.abs_diff(y));
        den = den.saturating_add(y);
    }
    (num, den)
}

/// E-S1: `D(t1, t2) = Σ_s |n_t1(s) − n_t2(s)| / Σ_s n_t2(s)`.
///
/// Syscalls are keyed by name, so the result is independent of the order
/// in which either tracer reported them. `None` when the reference run
/// (`t2`) saw no syscalls: the ratio is undefined, not zero.
#[must_use]
pub fn syscall_divergence(t1: &BTreeMap<String, u64>, t2: &BTreeMap<String, u64>) -> Option<f64> {
    let keys: BTreeSet<&String> = t1.keys().chain(t2.keys()).collect();
    let a: Vec<u64> = keys
        .iter()
        .map(|k| t1.get(*k).copied().unwrap_or(0))
        .collect();
    let b: Vec<u64> = keys
        .iter()
        .map(|k| t2.get(*k).copied().unwrap_or(0))
        .collect();
    let (num, den) = divergence_parts(&a, &b);
    (den > 0).then(|| num as f64 / den as f64)
}

/// Median of finite values; `None` on an empty or non-finite sample.
#[must_use]
pub fn median(xs: &[f64]) -> Option<f64> {
    let v = finite_sorted(xs)?;
    let n = v.len();
    let mid = n / 2;
    if n % 2 == 1 {
        Some(v[mid])
    } else {
        Some((v[mid - 1] + v[mid]) / 2.0)
    }
}

/// Nearest-rank quantile `q ∈ (0, 1]`; `None` on an empty or non-finite sample.
#[must_use]
pub fn quantile(xs: &[f64], q: f64) -> Option<f64> {
    if !(q > 0.0 && q <= 1.0) {
        return None;
    }
    let v = finite_sorted(xs)?;
    let rank = (q * v.len() as f64).ceil() as usize;
    v.get(rank.max(1) - 1).copied()
}

fn finite_sorted(xs: &[f64]) -> Option<Vec<f64>> {
    if xs.is_empty() || xs.iter().any(|x| !x.is_finite()) {
        return None;
    }
    let mut v = xs.to_vec();
    v.sort_by(f64::total_cmp);
    Some(v)
}

/// E-S1 gate: `med D(renacer, strace) ≤ q0.95(D(strace, strace))`.
///
/// `cross` holds one D per interleaved (ABBA) renacer/strace pair; `noise`
/// holds strace-vs-strace divergences. `None` when either sample is empty
/// or non-finite — an unmeasured gate is not a pass.
#[must_use]
pub fn parity_ok(cross: &[f64], noise: &[f64]) -> Option<bool> {
    Some(median(cross)? <= quantile(noise, 0.95)?)
}

/// E-S2: `gap_effect = ln(ρ_renacer / ρ_strace)` with `ρ = wall_traced / wall_untraced`.
///
/// Below 0 means renacer is cheaper. `None` unless both ratios are finite
/// and positive.
#[must_use]
pub fn overhead_gap_effect(rho_renacer: f64, rho_strace: f64) -> Option<f64> {
    let ok = |r: f64| r.is_finite() && r > 0.0;
    (ok(rho_renacer) && ok(rho_strace)).then(|| (rho_renacer / rho_strace).ln())
}

/// §2.5 per-class tolerance from `N` baseline counts of one syscall class.
///
/// `θ = max(ln(1 + 1/(1 + c̃)), 3σ̂)`, `σ̂ = 1.4826 · MAD(ln(1 + c))`.
/// Returns `(ℓ̃, θ)`; `None` on an empty baseline.
#[must_use]
pub fn golden_theta(baseline: &[u64]) -> Option<(f64, f64)> {
    let logs: Vec<f64> = baseline.iter().map(|&c| (c as f64).ln_1p()).collect();
    let ell = median(&logs)?;
    let dev: Vec<f64> = logs.iter().map(|l| (l - ell).abs()).collect();
    let sigma = 1.4826 * median(&dev)?;
    let c_med = ell.exp_m1();
    let floor = (1.0 / (1.0 + c_med)).ln_1p();
    Some((ell, floor.max(3.0 * sigma)))
}

/// Single-run golden-trace verdict (§2.5).
#[derive(Debug, Clone, PartialEq)]
pub enum GoldenVerdict {
    /// Every class within θ.
    Green,
    /// A class absent from the baseline appeared — always RED.
    NewClass(String),
    /// `|ln(1 + c) − ℓ̃| > 2θ` on one run.
    Breach {
        /// Syscall class.
        syscall: String,
        /// Observed count.
        count: u64,
    },
    /// `θ < |ln(1 + c) − ℓ̃| ≤ 2θ`: RED only if repeated on the next run.
    Watch(String),
    /// A baselined class has no baseline runs — the gate cannot decide.
    Unbaselined(String),
}

impl GoldenVerdict {
    /// RED on this run alone.
    #[must_use]
    pub fn is_red(&self) -> bool {
        matches!(
            self,
            Self::NewClass(_) | Self::Breach { .. } | Self::Unbaselined(_)
        )
    }
}

/// Compare one run against a per-class baseline (N runs per class).
///
/// Classes are checked in name order; the first RED wins, then the first
/// Watch. A baselined class missing from the run counts as 0.
#[must_use]
pub fn golden_verdict(
    baseline: &BTreeMap<String, Vec<u64>>,
    run: &BTreeMap<String, u64>,
) -> GoldenVerdict {
    if let Some(new) = run.keys().find(|k| !baseline.contains_key(*k)) {
        return GoldenVerdict::NewClass(new.clone());
    }
    let mut watch = None;
    for (class, counts) in baseline {
        let Some((ell, theta)) = golden_theta(counts) else {
            return GoldenVerdict::Unbaselined(class.clone());
        };
        let count = run.get(class).copied().unwrap_or(0);
        let dist = ((count as f64).ln_1p() - ell).abs();
        if dist > 2.0 * theta {
            return GoldenVerdict::Breach {
                syscall: class.clone(),
                count,
            };
        }
        if dist > theta && watch.is_none() {
            watch = Some(class.clone());
        }
    }
    watch.map_or(GoldenVerdict::Green, GoldenVerdict::Watch)
}

/// E-S3 integer core over kernel-set bitmasks: `(|A ∩ B|, |A ∪ B|)`.
#[must_use]
pub fn jaccard_parts(a: u64, b: u64) -> (u32, u32) {
    ((a & b).count_ones(), (a | b).count_ones())
}

/// E-S3: `J = |K_cgp ∩ K_nsys| / |K_cgp ∪ K_nsys|`.
///
/// Two empty sets are equal, so `J = 1`.
#[must_use]
pub fn kernel_jaccard(cgp: &BTreeSet<String>, nsys: &BTreeSet<String>) -> f64 {
    let inter = cgp.intersection(nsys).count();
    let union = cgp.union(nsys).count();
    if union == 0 {
        1.0
    } else {
        inter as f64 / union as f64
    }
}

/// E-S3 time bound: `J = 1` and `∀k: |ln(τ_cgp/τ_nsys)| ≤ q0.95(nsys self-noise)`.
///
/// `noise` holds `|ln(τ_nsys,1 / τ_nsys,2)|` over kernels of two nsys runs.
/// `None` when the noise sample is unusable or a time is non-positive.
#[must_use]
pub fn kernel_times_ok(
    cgp: &BTreeMap<String, f64>,
    nsys: &BTreeMap<String, f64>,
    noise: &[f64],
) -> Option<bool> {
    let bound = quantile(noise, 0.95)?;
    if cgp.keys().ne(nsys.keys()) {
        return Some(false);
    }
    for ((_, &tc), (_, &tn)) in cgp.iter().zip(nsys.iter()) {
        if !(tc > 0.0 && tn > 0.0 && tc.is_finite() && tn.is_finite()) {
            return None;
        }
        if (tc / tn).ln().abs() > bound {
            return Some(false);
        }
    }
    Some(true)
}

/// Provenance of a per-op breakdown (TRACE-001 §2.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provenance {
    /// Components were measured.
    Measured,
    /// Only the wall total was measured.
    WallClockTotal,
    /// Nothing below the total was measured.
    NotInstrumented,
}

/// Why an op-attribution breakdown is rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttributionDefect {
    /// `Σ components > wall`: time attributed that was never spent.
    SumExceedsWall {
        /// Σ measured components.
        sum_us: u64,
        /// Wall time.
        wall_us: u64,
    },
    /// A component is set under a provenance that measured none (R-1).
    ComponentUnderUnmeasured(usize),
    /// `Measured` with no component actually measured.
    MeasuredWithoutComponents,
}

/// `crux-F-25` check: Σ components ≤ wall, and unmeasured means `None`.
///
/// # Errors
///
/// Returns the first [`AttributionDefect`] found.
pub fn op_attribution_verdict(
    provenance: Provenance,
    components_us: &[Option<u64>],
    wall_us: u64,
) -> Result<(), AttributionDefect> {
    if provenance != Provenance::Measured {
        if let Some(i) = components_us.iter().position(Option::is_some) {
            return Err(AttributionDefect::ComponentUnderUnmeasured(i));
        }
        return Ok(());
    }
    if components_us.iter().all(Option::is_none) {
        return Err(AttributionDefect::MeasuredWithoutComponents);
    }
    let sum_us = components_us
        .iter()
        .flatten()
        .fold(0u64, |s, &c| s.saturating_add(c));
    if sum_us > wall_us {
        return Err(AttributionDefect::SumExceedsWall { sum_us, wall_us });
    }
    Ok(())
}

#[cfg(test)]
#[path = "crux_trace_tests.rs"]
mod tests;

#[cfg(kani)]
mod kani_proofs {
    use super::{divergence_parts, jaccard_parts};

    /// KANI-CRUX-D: the numerator is 0 exactly on identical vectors (so
    /// D ≥ 0 and D(t,t) = 0), and D is invariant to syscall ordering.
    #[kani::proof]
    #[kani::unwind(5)]
    fn verify_crux_d_identity_and_order() {
        let a: [u64; 4] = [kani::any(), kani::any(), kani::any(), kani::any()];
        let b: [u64; 4] = [kani::any(), kani::any(), kani::any(), kani::any()];
        for x in a.iter().chain(b.iter()) {
            kani::assume(*x < (1 << 32));
        }
        let (num, den) = divergence_parts(&a, &b);
        assert_eq!(num == 0, a == b, "KANI-CRUX-D: num = 0 iff identical");
        let (self_num, _) = divergence_parts(&a, &a);
        assert_eq!(self_num, 0, "KANI-CRUX-D: D(t,t) = 0");
        let i: usize = kani::any();
        let j: usize = kani::any();
        kani::assume(i < 4 && j < 4);
        let (mut pa, mut pb) = (a, b);
        pa.swap(i, j);
        pb.swap(i, j);
        assert_eq!(
            divergence_parts(&pa, &pb),
            (num, den),
            "KANI-CRUX-D: order invariant"
        );
    }

    /// KANI-CRUX-J: J ∈ [0,1] (inter ≤ union) and J = 1 iff the sets are equal.
    #[kani::proof]
    fn verify_crux_j_bounds_and_equality() {
        let a: u64 = kani::any();
        let b: u64 = kani::any();
        let (inter, union) = jaccard_parts(a, b);
        assert!(inter <= union, "KANI-CRUX-J: J <= 1");
        let j_is_one = union == 0 || inter == union;
        assert_eq!(j_is_one, a == b, "KANI-CRUX-J: J = 1 iff equal");
    }
}
