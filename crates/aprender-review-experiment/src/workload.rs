//! Review workload profile (PRM-001 v3 PRM-S2, §0.2 and §2.4; contract
//! `workload-profile-v1`, FALSIFY-WLP-001..004).
//!
//! The profile reads `agent-trace-v1` rows and reports, per lane, the
//! distribution of `tokens.input` and `tokens.output`. It decides §0.2's claim
//! that review is prefill-heavy (2k–32k tokens in, a short verdict out) and
//! sets the replay strata weights over distinct diffs.

use std::collections::BTreeMap;

use serde_json::{json, Value};

use crate::stats::percentile;

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
        self.input_p50 >= STRATA[0] as f64 && self.input_p95 <= STRATA[3] as f64
    }

    /// §0.2 "prefill-heavy": the long output is shorter than the typical input.
    #[must_use]
    pub fn prefill_heavy(&self) -> bool {
        self.output_p95 < self.input_p50
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
    let mut by_lane: BTreeMap<String, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
    let mut size: BTreeMap<String, u64> = BTreeMap::new();
    let mut excluded = 0;
    for line in rows.lines().filter(|l| !l.trim().is_empty()) {
        let Some((lane, diff, input, output)) = measured(line) else {
            excluded += 1;
            continue;
        };
        let e = by_lane.entry(lane).or_default();
        e.0.push(input as f64);
        e.1.push(output as f64);
        let s = size.entry(diff).or_insert(0);
        *s = (*s).max(input);
    }
    let lanes: Vec<LaneProfile> = by_lane
        .into_iter()
        .map(|(lane, (i, o))| LaneProfile {
            lane,
            n: i.len(),
            input_p50: percentile(&i, 0.5),
            input_p95: percentile(&i, 0.95),
            output_p50: percentile(&o, 0.5),
            output_p95: percentile(&o, 0.95),
        })
        .collect();
    let (claim, refusals) = decide(&lanes, min_n);
    let over_32k = size.values().filter(|&&s| s > STRATA[3]).count();
    let in_range = size.len() - over_32k;
    let strata = STRATA
        .iter()
        .enumerate()
        .map(|(k, &b)| {
            let lo = if k == 0 { 0 } else { STRATA[k - 1] };
            let c = size.values().filter(|&&s| s > lo && s <= b).count();
            let w = if in_range == 0 {
                0.0
            } else {
                c as f64 / in_range as f64
            };
            (b, w)
        })
        .collect();
    Profile {
        lanes,
        excluded,
        units: size.len(),
        over_32k,
        strata,
        claim,
        refusals,
    }
}

/// `(lane, diff_sha256, input, output)` of a measured row; `None` for a failed
/// capture, a line that is not a row, or a row without positive input and an
/// output count.
fn measured(line: &str) -> Option<(String, String, u64, u64)> {
    let v: Value = serde_json::from_str(line).ok()?;
    if v["trace_status"].as_str() == Some("capture_failed") {
        return None;
    }
    let input = v["tokens"]["input"].as_u64().filter(|&n| n > 0)?;
    let output = v["tokens"]["output"].as_u64()?;
    Some((
        v["lane"].as_str()?.to_string(),
        v["diff_sha256"].as_str()?.to_string(),
        input,
        output,
    ))
}

fn decide(lanes: &[LaneProfile], min_n: usize) -> (Option<bool>, Vec<String>) {
    let deciding: Vec<&LaneProfile> = lanes.iter().filter(|l| l.n >= min_n).collect();
    if deciding.is_empty() {
        return (
            None,
            vec![format!(
                "S-14 §0.2 undecided: no lane has min_n = {min_n} measured rows"
            )],
        );
    }
    let mut refusals = Vec::new();
    for l in deciding {
        if !l.in_band() {
            refusals.push(format!(
                "§0.2 falsified by {}: input p50 {} / p95 {} outside 2000–32000",
                l.lane, l.input_p50, l.input_p95
            ));
        }
        if !l.prefill_heavy() {
            refusals.push(format!(
                "§0.2 falsified by {}: output p95 {} ≥ input p50 {}",
                l.lane, l.output_p95, l.input_p50
            ));
        }
    }
    (Some(refusals.is_empty()), refusals)
}

/// The receipt's `workload` block (§7 template) plus the claim, strata and
/// exclusions it was decided on.
#[must_use]
#[allow(clippy::disallowed_methods)] // `json!` expands to an `unwrap` of an infallible `to_value`
pub fn render(p: &Profile) -> Value {
    let pair = |f: fn(&LaneProfile) -> [f64; 2]| -> serde_json::Map<String, Value> {
        p.lanes
            .iter()
            .map(|l| (l.lane.clone(), json!(f(l))))
            .collect()
    };
    let strata: serde_json::Map<String, Value> = p
        .strata
        .iter()
        .map(|(b, w)| (format!("{}k", b / 1000), json!(w)))
        .collect();
    let n: serde_json::Map<String, Value> = p
        .lanes
        .iter()
        .map(|l| (l.lane.clone(), json!(l.n)))
        .collect();
    json!({"workload": {
        "input_tokens_p50_p95": pair(|l| [l.input_p50, l.input_p95]),
        "output_tokens_p50_p95": pair(|l| [l.output_p50, l.output_p95]),
        "n": n,
        "claim_0_2": p.claim,
        "refusals": p.refusals,
        "strata": strata,
        "units": p.units,
        "over_32k": p.over_32k,
        "excluded": p.excluded,
    }})
}

#[cfg(test)]
#[path = "workload_tests.rs"]
mod tests;
