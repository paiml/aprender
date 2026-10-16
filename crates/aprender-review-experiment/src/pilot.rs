//! REX-05 pilot projection: the variance and runtime of the dev-item pilot on
//! one admitted cell, and the §2.2 sample-size rule applied to it.
//!
//! The rule (analysis plan §Sample-size rule): project the test-split Wilson
//! half-width on recall at the pilot's point estimate; above
//! [`MAX_HALF_WIDTH`] the corpus grows, and the hypotheses do not change.
//! A pilot with no defect item in its denominator has no point estimate, so
//! the rule falls back to the §2.2 design basis p = 0.5 and says so.

use crate::score::Ratio;
use crate::stats::{percentile, sample_sd, wilson_half_width};
use serde::Serialize;

/// §2.2: grow the corpus when the projected half-width exceeds this.
pub const MAX_HALF_WIDTH: f64 = 0.12;

/// Where the projected recall came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Basis {
    /// The pilot's own recall point estimate.
    Pilot,
    /// No defect item in the pilot denominator: the §2.2 design p = 0.5.
    DesignWorstCase,
}

/// Spread of one timing sample, in milliseconds.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Spread {
    pub n: usize,
    pub mean: f64,
    /// `None` below two values.
    pub sd: Option<f64>,
    pub p50: f64,
    pub p95: f64,
}

impl Spread {
    /// `None` for an empty sample (nothing ran, so there is no spread).
    #[must_use]
    pub fn of(x: &[f64]) -> Option<Self> {
        if x.is_empty() || x.iter().any(|v| !v.is_finite()) {
            return None;
        }
        Some(Self {
            n: x.len(),
            mean: x.iter().sum::<f64>() / x.len() as f64,
            sd: sample_sd(x),
            p50: percentile(x, 0.50),
            p95: percentile(x, 0.95),
        })
    }
}

/// What the pilot projects for the full run on this cell.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Projection {
    pub recall: Ratio,
    pub basis: Basis,
    pub p: f64,
    pub n_defect_test: u64,
    /// Wilson 95 % half-width at `p` on `n_defect_test` (`None` when n = 0).
    pub half_width: Option<f64>,
    /// §2.2: true when `half_width` exceeds [`MAX_HALF_WIDTH`] or cannot be
    /// computed. An unknown width is never a reason to keep the corpus.
    pub grow_corpus: bool,
    pub warm: Option<Spread>,
    pub cold: Option<Spread>,
    /// Test items plus the 10 % determinism re-run.
    pub projected_requests: u64,
    /// `projected_requests` × warm mean; `None` when nothing warm ran.
    pub projected_runtime_s: Option<f64>,
}

/// Apply the §2.2 rule and the runtime projection to one pilot.
#[must_use]
pub fn project(
    recall: Ratio,
    warm_wall_ms: &[f64],
    cold_wall_ms: &[f64],
    n_defect_test: u64,
    n_test: u64,
) -> Projection {
    let (basis, p) = match recall.value() {
        Some(p) => (Basis::Pilot, p),
        None => (Basis::DesignWorstCase, 0.5),
    };
    let k = (p * n_defect_test as f64).round() as u64;
    let half_width = wilson_half_width(k, n_defect_test);
    let warm = Spread::of(warm_wall_ms);
    let projected_requests = n_test + n_test.div_ceil(10);
    Projection {
        recall,
        basis,
        p,
        n_defect_test,
        half_width,
        grow_corpus: half_width.is_none_or(|h| h > MAX_HALF_WIDTH),
        warm,
        cold: Spread::of(cold_wall_ms),
        projected_requests,
        projected_runtime_s: warm.map(|w| w.mean * projected_requests as f64 / 1e3),
    }
}

#[cfg(test)]
#[path = "pilot_tests.rs"]
mod tests;
