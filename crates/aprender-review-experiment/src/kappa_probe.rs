//! PRM-C13 (was PRA-001 T13): the lane κ_err probe and its independence gate
//! (PRM-001 v3 §3 H7 and S-14; contract
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
//! The gate is H7 as PRM-001 v3 §3 states it, δ-free: on the items the shadow
//! lane and EVERY counted lane scored, κ_err(shadow, V) ≤ the largest
//! voter–voter κ_err for each voter V, decided by the prereg-locked
//! [`crate::stats::h7_holds`]. It refuses (S-14) below `min_n` common items,
//! on a voter with no rows, and when `h7_holds` is undecided. The voter set and
//! `min_n` come only from the run manifest, registered strictly before
//! training started.

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

type Items<'a> = BTreeMap<(&'a str, u64), BTreeMap<&'a str, Vec<&'a Row>>>;

/// Matured gold `split` rows grouped item → lane → rows.
fn items_of<'a>(rows: &'a [Row], split: &str) -> Items<'a> {
    let mut items: Items<'a> = BTreeMap::new();
    for r in rows
        .iter()
        .filter(|r| r.gold_matured && r.split == split && r.defect.is_some())
    {
        items
            .entry((&r.quorum_id, r.round))
            .or_default()
            .entry(&r.lane)
            .or_default()
            .push(r);
    }
    items
}

/// A lane's `(flag, defect)` on one item; `None` if absent, unparsed, or
/// ambiguous (two rows for one item).
fn one(m: &BTreeMap<&str, Vec<&Row>>, l: &str) -> Option<(bool, bool)> {
    match m.get(l).map(Vec::as_slice) {
        Some([r]) => Some((r.flag?, r.defect?)),
        _ => None,
    }
}

/// κ_err of `shadow` against every counted lane, on matured gold `split` rows.
#[must_use]
pub fn probe(rows: &[Row], split: &str, shadow: &str, min_n: usize) -> Scoreboard {
    let items = items_of(rows, split);
    let counted: BTreeSet<&str> = rows
        .iter()
        .filter(|r| r.gold_matured && r.split == split && r.defect.is_some())
        .filter(|r| r.counted && r.lane != shadow)
        .map(|r| r.lane.as_str())
        .collect();
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

/// The `lane_independence` block of a run manifest. H7 is δ-free (PRM-001 v3
/// §3), so what is pre-registered is the voter set and the item floor.
#[derive(Debug, Clone, Serialize)]
pub struct Manifest {
    /// Shared-item floor for the gate to decide.
    pub min_n: usize,
    /// The counted lanes (voters) the gate must see on every item.
    pub counted_lanes: Vec<String>,
    /// When the block was registered (`YYYY-MM-DDTHH:MM:SSZ`).
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

/// Parse and validate a run manifest's `lane_independence` block. A missing
/// or unusable field, a block registered after training started, or a `delta`
/// (v2's rule, gone in v3: nothing would read it) is an error.
pub fn parse_manifest(json: &str) -> Result<Manifest, String> {
    let v: Value = serde_json::from_str(json).map_err(|e| format!("manifest: {e}"))?;
    let li = v
        .get("lane_independence")
        .ok_or("manifest has no lane_independence block")?;
    if li.get("delta").is_some() {
        return Err(
            "lane_independence.delta: H7 is δ-free in PRM-001 v3 (κ(qwen, V) ≤ max voter–voter κ); remove it"
                .into(),
        );
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
            "lane_independence registered at {registered_at}, not before training started at {training_started_at}"
        ));
    }
    Ok(Manifest {
        min_n: usize::try_from(min_n).map_err(|e| e.to_string())?,
        counted_lanes,
        registered_at,
        training_started_at,
    })
}

/// One lane pair's κ_err on the gate's common items.
#[derive(Debug, Clone, Serialize)]
pub struct GatePair {
    pub pair: [String; 2],
    pub kappa_err: Option<f64>,
}

/// The gate's decision and the κ it decided on.
#[derive(Debug, Clone, Serialize)]
pub struct GateVerdict {
    /// `true` only when [`crate::stats::h7_holds`] returned `Some(true)` on at
    /// least `min_n` common items.
    pub pass: bool,
    /// Every reason for refusal; empty when `pass`.
    pub refusals: Vec<String>,
    /// Items every lane (shadow and each voter) scored, gold agreeing.
    pub n: usize,
    /// κ_err(shadow, V) per voter, manifest order.
    pub qwen_vs_voter: Vec<GatePair>,
    /// κ_err(V_i, V_j) per voter pair, i < j.
    pub voter_vs_voter: Vec<GatePair>,
    /// The largest voter–voter κ_err, when every pair is defined.
    pub ceiling: Option<f64>,
}

/// H7 (PRM-001 v3 §3, δ-free) on candidate rows: with I the matured gold
/// `split` items that `shadow` and EVERY counted lane scored (one parsed row
/// each, gold agreeing), `pass` ⇔ `stats::h7_holds` on I is `Some(true)`, i.e.
/// κ_err(shadow, V) ≤ max voter–voter κ_err for every voter V. Refuses (S-14)
/// on fewer than `min_n` common items, a voter with no row, and an undecided
/// `h7_holds` (fewer than two voters, or an undefined κ).
#[must_use]
pub fn gate(rows: &[Row], split: &str, shadow: &str, m: &Manifest) -> GateVerdict {
    let items = items_of(rows, split);
    let mut refusals = Vec::new();
    let voters: Vec<&str> = m.counted_lanes.iter().map(String::as_str).collect();
    for v in &voters {
        if !items.values().any(|l| l.contains_key(v)) {
            refusals.push(format!("S-14 {v}: no scored rows"));
        }
    }
    // err[lane] over I, in item order.
    let lanes: Vec<&str> = std::iter::once(shadow)
        .chain(voters.iter().copied())
        .collect();
    let mut err: Vec<Vec<bool>> = vec![Vec::new(); lanes.len()];
    for m in items.values() {
        let Some(got) = lanes.iter().map(|l| one(m, l)).collect::<Option<Vec<_>>>() else {
            continue;
        };
        // All rows describe one PR; disagreeing gold is not gold.
        if got.iter().all(|&(_, d)| d == got[0].1) {
            for (e, (f, d)) in err.iter_mut().zip(got) {
                e.push(f != d);
            }
        }
    }
    let n = err[0].len();
    if n < m.min_n {
        refusals.push(format!(
            "S-14 insufficient: {n} items scored by the shadow lane and every voter, min_n={}",
            m.min_n
        ));
    }
    let k = |i: usize, j: usize| GatePair {
        pair: [lanes[i].to_string(), lanes[j].to_string()],
        kappa_err: crate::stats::error_kappa(&err[i], &err[j]),
    };
    let qwen_vs_voter: Vec<GatePair> = (1..lanes.len()).map(|j| k(0, j)).collect();
    let voter_vs_voter: Vec<GatePair> = (1..lanes.len())
        .flat_map(|i| (i + 1..lanes.len()).map(move |j| (i, j)))
        .map(|(i, j)| k(i, j))
        .collect();
    let kappas = |p: &[GatePair]| p.iter().map(|p| p.kappa_err).collect::<Vec<_>>();
    let ceiling = kappas(&voter_vs_voter)
        .into_iter()
        .collect::<Option<Vec<f64>>>()
        .and_then(|v| v.into_iter().reduce(f64::max));
    match crate::stats::h7_holds(&kappas(&qwen_vs_voter), &kappas(&voter_vs_voter)) {
        Some(true) => {}
        Some(false) => {
            let c = ceiling.unwrap_or(f64::NAN);
            for p in &qwen_vs_voter {
                if let Some(q) = p.kappa_err.filter(|&q| q > c) {
                    refusals.push(format!(
                        "S-14 {}: κ_err {q:.4} > voter–voter max {c:.4}",
                        p.pair[1]
                    ));
                }
            }
        }
        None => refusals
            .push("S-14 H7 undecided: fewer than two voters, or a κ_err is undefined".into()),
    }
    GateVerdict {
        pass: refusals.is_empty(),
        refusals,
        n,
        qwen_vs_voter,
        voter_vs_voter,
        ceiling,
    }
}

#[cfg(test)]
#[path = "kappa_probe_tests.rs"]
mod tests;
