//! Review workload profile (PRM-001 v3 PRM-S2, §0.2 and §2.4; contract
//! `workload-profile-v1`, FALSIFY-WLP-001..004).
//!
//! The profile reads `agent-trace-v1` rows and reports, per lane, the
//! distribution of `tokens.input` and `tokens.output`. It decides §0.2's claim
//! that review is prefill-heavy (2k–32k tokens in, a short verdict out) and
//! sets the replay strata weights over distinct diffs.

use std::collections::BTreeMap;

use serde_json::{json, Value};

/// §2.4 replay strata upper bounds, in input tokens.
pub const STRATA: [u64; 4] = [2000, 8000, 16000, 32000];

/// Per-lane token quantiles over the measured rows.
#[derive(Debug, Clone, PartialEq)]
pub struct LaneProfile {
    pub lane: String,
    pub n: usize,
    pub input_p50: f64,
    pub input_p95: f64,
    pub output_p50: f64,
    pub output_p95: f64,
}

impl LaneProfile {
    /// §0.2 "a diff of 2k–32k tokens goes in".
    #[must_use]
    pub fn in_band(&self) -> bool {
        false
    }

    /// §0.2 "prefill-heavy": the long output is shorter than the typical input.
    #[must_use]
    pub fn prefill_heavy(&self) -> bool {
        false
    }
}

/// The profile of one set of trace rows.
#[derive(Debug, Clone, PartialEq)]
pub struct Profile {
    pub lanes: Vec<LaneProfile>,
    /// Rows that could not be measured (capture failed, unparsed, no tokens).
    pub excluded: usize,
    /// Distinct `diff_sha256` work units with a measurement.
    pub units: usize,
    /// Units over the top stratum; not weighted.
    pub over_32k: usize,
    /// Weight per stratum bound, over the in-range units.
    pub strata: Vec<(u64, f64)>,
    /// §0.2: `None` when no lane reached `min_n`.
    pub claim: Option<bool>,
    /// Why the claim is not confirmed (S-code lines), empty when it is.
    pub refusals: Vec<String>,
}

/// Build the profile of JSONL `rows`; lanes with fewer than `min_n` measured
/// rows are reported but do not decide the claim.
#[must_use]
pub fn profile(rows: &str, min_n: usize) -> Profile {
    let _ = (rows, min_n);
    Profile {
        lanes: Vec::new(),
        excluded: 0,
        units: 0,
        over_32k: 0,
        strata: Vec::new(),
        claim: Some(true),
        refusals: Vec::new(),
    }
}

/// The receipt's `workload` block (§7 template) plus the claim, strata and
/// exclusions it was decided on.
#[must_use]
pub fn render(p: &Profile) -> Value {
    let _ = p;
    json!({})
}

#[allow(dead_code)]
fn unused(_: BTreeMap<String, ()>) {}

#[cfg(test)]
#[path = "workload_tests.rs"]
mod tests;
