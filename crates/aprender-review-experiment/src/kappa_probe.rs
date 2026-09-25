//! PRM-C13 (was PRA-001 T13): the lane κ_err probe and its independence gate
//! (PRM-001 v2 §5.4 H7 and S-14; contract
//! `lane-independence-v1`).
//!
//! κ_err is Cohen's κ on binary per-item error indicators against GOLD, for
//! every (shadow lane, counted lane) pair, over matured gold rows of one split.
//! It reuses the prereg-locked [`crate::stats::error_kappa`]; this module only
//! decides which items count and what an error is.
//!
//! - An item is one `(quorum_id, round)` whose rows are `label_tier = gold`,
//!   matured (`outcome_matured_at` non-null), on the requested split, with
//!   `trace_status = ok` and a settled outcome: `merged` is clean,
//!   `reverted_le14d` / `regression_escape` is a defect. `pending` and
//!   `closed_unmerged` carry no gold label and are excluded.
//! - A lane flags a defect with `request_changes` and clears it with `approve`.
//!   Any other verdict, a `failed` parse or a non-`Verdict` state is UNPARSED:
//!   the item is excluded for that pair, never counted as an error.
//! - A pair with fewer than `min_n` shared items, or with κ undefined, is
//!   `insufficient`: it never prints a number (S-14).
//!
//! The gate refuses a candidate when κ_err(candidate, L) − baseline(L) > δ for
//! any counted lane L, or when any counted lane is insufficient or missing on
//! either side. δ comes only from the run manifest, registered strictly before
//! training started; nothing here defaults or widens it.

use serde::Serialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

/// The fields of one agent-trace-v1 row the probe reads.
#[derive(Debug, Clone)]
pub struct Row {
    /// `quorum_id`
    pub quorum_id: String,
    /// `round`
    pub round: u64,
    /// `lane`
    pub lane: String,
    /// `counted`
    pub counted: bool,
    /// `true` = flagged a defect, `false` = cleared it, `None` = unparsed.
    pub flag: Option<bool>,
    /// `true` = gold defect, `false` = clean, `None` = no gold label.
    pub defect: Option<bool>,
    /// Gold, matured, `trace_status = ok`.
    pub gold_matured: bool,
    /// `split_guard.split`
    pub split: String,
}

fn text<'a>(v: &'a Value, k: &str) -> &'a str {
    v.get(k).and_then(Value::as_str).unwrap_or("")
}

fn row_of(v: &Value) -> Result<Row, String> {
    let quorum_id = v
        .get("quorum_id")
        .and_then(Value::as_str)
        .ok_or("no quorum_id")?
        .to_string();
    let round = v.get("round").and_then(Value::as_u64).ok_or("no round")?;
    let lane = v
        .get("lane")
        .and_then(Value::as_str)
        .ok_or("no lane")?
        .to_string();
    let parsed =
        text(v, "lane_state") == "Verdict" && matches!(text(v, "parse_status"), "ok" | "repaired");
    let flag = match (parsed, text(v, "verdict")) {
        (true, "request_changes") => Some(true),
        (true, "approve") => Some(false),
        _ => None,
    };
    let defect = match text(v, "outcome") {
        "merged" => Some(false),
        "reverted_le14d" | "regression_escape" => Some(true),
        _ => None,
    };
    let matured = v.get("outcome_matured_at").is_some_and(|m| !m.is_null());
    Ok(Row {
        quorum_id,
        round,
        lane,
        counted: v.get("counted").and_then(Value::as_bool).unwrap_or(false),
        flag,
        defect,
        gold_matured: text(v, "label_tier") == "gold" && matured && text(v, "trace_status") == "ok",
        split: v
            .get("split_guard")
            .map(|s| text(s, "split"))
            .unwrap_or("")
            .to_string(),
    })
}

/// Parse agent-trace-v1 JSONL. A malformed line is an error naming its line
/// number, never a silently dropped row.
pub fn parse_rows(jsonl: &str) -> Result<Vec<Row>, String> {
    jsonl
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(i, l)| {
            serde_json::from_str::<Value>(l)
                .map_err(|e| e.to_string())
                .and_then(|v| row_of(&v))
                .map_err(|e| format!("line {}: {e}", i + 1))
        })
        .collect()
}

/// Whether a pair produced a κ.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PairStatus {
    /// κ_err is defined on at least `min_n` shared items.
    Ok,
    /// Too few shared items, or κ undefined (S-14).
    Insufficient,
}

/// κ_err of one (shadow lane, counted lane) pair.
#[derive(Debug, Clone, Serialize)]
pub struct PairKappa {
    /// The counted lane.
    pub counted_lane: String,
    /// Shared scored items.
    pub n: usize,
    /// `None` unless `status` is `Ok`.
    pub kappa_err: Option<f64>,
    /// See [`PairStatus`].
    pub status: PairStatus,
}

/// Every pair's κ_err for one shadow lane on one split.
#[derive(Debug, Clone, Serialize)]
pub struct Scoreboard {
    /// Always `lane-independence-v1`.
    pub schema: &'static str,
    /// The split scored.
    pub split: String,
    /// The shadow lane (the candidate side of every pair).
    pub shadow_lane: String,
    /// The shared-item floor applied.
    pub min_n: usize,
    /// One entry per counted lane present in the rows, sorted by lane.
    pub pairs: Vec<PairKappa>,
}

/// κ_err of `shadow` against every counted lane, on matured gold `split` rows.
#[must_use]
pub fn probe(rows: &[Row], split: &str, shadow: &str, min_n: usize) -> Scoreboard {
    // item -> lane -> rows; a lane with two rows for one item is ambiguous.
    let mut items: BTreeMap<(&str, u64), BTreeMap<&str, Vec<&Row>>> = BTreeMap::new();
    let mut counted: BTreeSet<&str> = BTreeSet::new();
    for r in rows
        .iter()
        .filter(|r| r.gold_matured && r.split == split && r.defect.is_some())
    {
        if r.counted && r.lane != shadow {
            counted.insert(&r.lane);
        }
        items
            .entry((&r.quorum_id, r.round))
            .or_default()
            .entry(&r.lane)
            .or_default()
            .push(r);
    }
    let one = |m: &BTreeMap<&str, Vec<&Row>>, l: &str| -> Option<(bool, bool)> {
        match m.get(l).map(Vec::as_slice) {
            Some([r]) => Some((r.flag?, r.defect?)),
            _ => None,
        }
    };
    let pairs = counted
        .into_iter()
        .map(|lane| {
            let (mut a, mut b) = (Vec::new(), Vec::new());
            for m in items.values() {
                if let (Some((fs, ds)), Some((fl, dl))) = (one(m, shadow), one(m, lane)) {
                    // Both rows describe one PR; disagreeing gold is not gold.
                    if ds == dl {
                        a.push(fs != ds);
                        b.push(fl != dl);
                    }
                }
            }
            let kappa_err = (a.len() >= min_n)
                .then(|| crate::stats::error_kappa(&a, &b))
                .flatten();
            PairKappa {
                counted_lane: lane.to_string(),
                n: a.len(),
                kappa_err,
                status: if kappa_err.is_some() {
                    PairStatus::Ok
                } else {
                    PairStatus::Insufficient
                },
            }
        })
        .collect();
    Scoreboard {
        schema: "lane-independence-v1",
        split: split.to_string(),
        shadow_lane: shadow.to_string(),
        min_n,
        pairs,
    }
}

/// The `lane_independence` block of a run manifest.
#[derive(Debug, Clone, Serialize)]
pub struct Manifest {
    /// Allowed κ_err rise over baseline, per counted lane.
    pub delta: f64,
    /// Shared-item floor for a pair to be scored.
    pub min_n: usize,
    /// The counted lanes the gate must see on both sides.
    pub counted_lanes: Vec<String>,
    /// When δ was registered (`YYYY-MM-DDTHH:MM:SSZ`).
    pub registered_at: String,
    /// When candidate training started (`YYYY-MM-DDTHH:MM:SSZ`).
    pub training_started_at: String,
}

/// Strict UTC `YYYY-MM-DDTHH:MM:SSZ`, so string order is time order.
fn utc_stamp(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() == 20
        && b.iter().enumerate().all(|(i, &c)| match i {
            4 | 7 => c == b'-',
            10 => c == b'T',
            13 | 16 => c == b':',
            19 => c == b'Z',
            _ => c.is_ascii_digit(),
        })
}

/// Parse and validate a run manifest's `lane_independence` block. Every way δ
/// could be missing, unusable or chosen after training started is an error.
pub fn parse_manifest(json: &str) -> Result<Manifest, String> {
    let v: Value = serde_json::from_str(json).map_err(|e| format!("manifest: {e}"))?;
    let li = v
        .get("lane_independence")
        .ok_or("manifest has no lane_independence block")?;
    let delta = li
        .get("delta")
        .and_then(Value::as_f64)
        .ok_or("lane_independence.delta is not registered")?;
    if !delta.is_finite() || delta < 0.0 {
        return Err(format!(
            "lane_independence.delta {delta} is not a finite δ ≥ 0"
        ));
    }
    let min_n = li
        .get("min_n")
        .and_then(Value::as_u64)
        .filter(|&n| n > 0)
        .ok_or("lane_independence.min_n must be a positive integer")?;
    let counted_lanes: Vec<String> = li
        .get("counted_lanes")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|l| l.as_str().map(str::to_string))
                .collect()
        })
        .filter(|a: &Vec<String>| !a.is_empty())
        .ok_or("lane_independence.counted_lanes must name at least one lane")?;
    let stamp = |k: &str| -> Result<String, String> {
        li.get(k)
            .and_then(Value::as_str)
            .filter(|s| utc_stamp(s))
            .map(str::to_string)
            .ok_or(format!(
                "lane_independence.{k} must be YYYY-MM-DDTHH:MM:SSZ"
            ))
    };
    let registered_at = stamp("registered_at")?;
    let training_started_at = stamp("training_started_at")?;
    if registered_at >= training_started_at {
        return Err(format!(
            "δ registered at {registered_at}, not before training started at {training_started_at}"
        ));
    }
    Ok(Manifest {
        delta,
        min_n: usize::try_from(min_n).map_err(|e| e.to_string())?,
        counted_lanes,
        registered_at,
        training_started_at,
    })
}

/// The gate's decision.
#[derive(Debug, Clone, Serialize)]
pub struct GateVerdict {
    /// `true` only when every counted lane is scored on both sides and none rose past δ.
    pub pass: bool,
    /// Every reason for refusal; empty when `pass`.
    pub refusals: Vec<String>,
}

/// Refuse the candidate if κ_err rose past δ for any counted lane, or if any
/// counted lane cannot be scored on either side (S-14). The comparison is
/// `candidate − baseline > δ`, so a rise of exactly δ passes.
pub fn gate(
    baseline: &Scoreboard,
    candidate: &Scoreboard,
    m: &Manifest,
) -> Result<GateVerdict, String> {
    let kappa = |b: &Scoreboard, lane: &str, side: &str| -> Result<f64, String> {
        let p = b
            .pairs
            .iter()
            .find(|p| p.counted_lane == lane)
            .ok_or_else(|| format!("S-14 {lane}: no {side} pair"))?;
        match p.kappa_err {
            Some(k) if p.n >= m.min_n => Ok(k),
            _ => Err(format!(
                "S-14 {lane}: {side} insufficient (n={}, min_n={})",
                p.n, m.min_n
            )),
        }
    };
    let mut refusals = Vec::new();
    for lane in &m.counted_lanes {
        match (
            kappa(baseline, lane, "baseline"),
            kappa(candidate, lane, "candidate"),
        ) {
            (Ok(k0), Ok(k1)) => {
                if k1 - k0 > m.delta {
                    refusals.push(format!(
                        "S-14 {lane}: κ_err {k1:.4} > baseline {k0:.4} + δ {}",
                        m.delta
                    ));
                }
            }
            (a, b) => refusals.extend([a, b].into_iter().filter_map(Result::err)),
        }
    }
    Ok(GateVerdict {
        pass: refusals.is_empty(),
        refusals,
    })
}

#[cfg(test)]
#[path = "kappa_probe_tests.rs"]
mod tests;
