//! EXT-19 (aprender#4401): the engine speed ledger (EXT-001 §3.7, goal G3).
//!
//! On every aprender tag the released binary is measured on each REX-08-ruled
//! cell against the blessed model, next to the EXT-28 llama.cpp arm. Each
//! (tag, cell) pair gets exactly one ledger row, one JSON object per line:
//! `measured` (apr's decode tok/s and every arm's, each arm carrying its EXT-26
//! comparator block) or `not_run` with a reason. Nothing else counts: G3 is
//! 100% coverage, and a missing pair is a hole, never a pass.
//!
//! The ratio apr / llama.cpp is computed HERE and nowhere else. No row stores
//! it — the row schema refuses a `ratio` field — and no published surface may
//! print it (T28: a comparator ratio is illegal regardless of citation). It is
//! the ratchet's input and nothing more.
//!
//! The ratchet is shrink-only on the gap and arms after [`ARM_AFTER`] measured
//! records. Its floor is the running max of the median of the last three
//! ratios before the judged one: one lucky run cannot raise it (the median
//! ignores it), a sustained improvement does, and it never comes back down. A
//! record below `floor × (1 − TOLERANCE)` is RED — FALSIFY-EXT-022's planted
//! sleep in the decode loop.

use super::comparator::{check_block, is_sha256, ComparatorBlock};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;

/// The arm every ratio is taken against (the existing oracle, EXT-001 C2).
pub(crate) const REFERENCE_ARM: &str = "llama.cpp";
/// Measured records that must precede a judged one before the ratchet arms.
pub(crate) const ARM_AFTER: usize = 3;
/// How far below the floor a record may land before it is RED.
pub(crate) const TOLERANCE: f64 = 0.05;

/// One competitor arm's speed on the cell.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ArmSpeed {
    pub arm: String,
    pub decode_tok_s: f64,
    pub comparator: ComparatorBlock,
}

/// A measured (tag, cell): apr's speed, the arms', and the receipt they came from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Measured {
    pub apr_decode_tok_s: f64,
    pub receipt_sha256: String,
    pub arms: Vec<ArmSpeed>,
}

/// A (tag, cell) that was not measured, and why.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NotRun {
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Outcome {
    Measured(Measured),
    NotRun(NotRun),
}

/// One ledger line.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Row {
    pub tag: String,
    pub cell: String,
    pub outcome: Outcome,
}

fn positive(what: &str, x: f64) -> Result<(), String> {
    if x.is_finite() && x > 0.0 {
        Ok(())
    } else {
        Err(format!("{what} must be a positive finite tok/s, got {x}"))
    }
}

fn check_measured(m: &Measured) -> Result<(), String> {
    positive("apr_decode_tok_s", m.apr_decode_tok_s)?;
    if !is_sha256(&m.receipt_sha256) {
        return Err("`receipt_sha256` is not 64 lowercase hex".into());
    }
    let mut seen = BTreeSet::new();
    for a in &m.arms {
        if !seen.insert(a.arm.as_str()) {
            return Err(format!("arm `{}` appears twice", a.arm));
        }
        positive(&format!("arm `{}` decode_tok_s", a.arm), a.decode_tok_s)?;
        let v = serde_json::to_value(&a.comparator).map_err(|e| e.to_string())?;
        check_block(&v).map_err(|e| format!("arm `{}`: {e}", a.arm))?;
    }
    if !seen.contains(REFERENCE_ARM) {
        return Err(format!("measured row has no `{REFERENCE_ARM}` arm"));
    }
    Ok(())
}

/// Parse and validate one ledger line.
pub(crate) fn parse_row(v: &Value) -> Result<Row, String> {
    let row: Row = serde_json::from_value(v.clone()).map_err(|e| format!("ledger row: {e}"))?;
    if row.tag.trim().is_empty() || row.cell.trim().is_empty() {
        return Err("ledger row: empty `tag` or `cell`".into());
    }
    match &row.outcome {
        Outcome::Measured(m) => check_measured(m)?,
        Outcome::NotRun(n) if n.reason.trim().is_empty() => {
            return Err("ledger row: `not_run` needs a reason".into());
        }
        Outcome::NotRun(_) => {}
    }
    Ok(row)
}

/// Parse a JSONL ledger. Blank lines are skipped; any bad line fails the
/// whole ledger, naming its 1-based line number.
pub(crate) fn parse_ledger(jsonl: &str) -> Result<Vec<Row>, String> {
    let mut rows = Vec::new();
    for (i, line) in jsonl.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(line).map_err(|e| format!("line {}: {e}", i + 1))?;
        rows.push(parse_row(&v).map_err(|e| format!("line {}: {e}", i + 1))?);
    }
    Ok(rows)
}

/// G3 coverage: the (tag, cell) pairs with no row. `not_run` covers its pair.
/// A pair with two rows is an error — the ledger is append-only, and a second
/// row would let a later run silently replace an earlier verdict.
pub(crate) fn uncovered(
    rows: &[Row],
    tags: &[String],
    cells: &[String],
) -> Result<Vec<(String, String)>, String> {
    let mut have = BTreeSet::new();
    for r in rows {
        if !have.insert((r.tag.as_str(), r.cell.as_str())) {
            return Err(format!("({}, {}) has two ledger rows", r.tag, r.cell));
        }
    }
    let mut missing = Vec::new();
    for t in tags {
        for c in cells {
            if !have.contains(&(t.as_str(), c.as_str())) {
                missing.push((t.clone(), c.clone()));
            }
        }
    }
    Ok(missing)
}

/// The ratchet verdict for the newest measured record of one cell.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Ratchet {
    /// Fewer than [`ARM_AFTER`] records precede the newest one.
    Unarmed { records: usize },
    Green { tag: String, ratio: f64, floor: f64 },
    Red { tag: String, ratio: f64, floor: f64 },
}

fn ratio(m: &Measured) -> Option<f64> {
    m.arms
        .iter()
        .find(|a| a.arm == REFERENCE_ARM)
        .map(|a| m.apr_decode_tok_s / a.decode_tok_s)
}

// The floor window is a median of three; changing ARM_AFTER changes that too.
const _: () = assert!(ARM_AFTER == 3);

fn median3(xs: &[f64]) -> f64 {
    let mut w = [xs[0], xs[1], xs[2]];
    w.sort_by(f64::total_cmp);
    w[1]
}

/// The (tag, ratio) series of one cell, in `tags` order (release order).
fn series<'a>(rows: &'a [Row], tags: &'a [String], cell: &str) -> Vec<(&'a str, f64)> {
    tags.iter()
        .filter_map(|t| {
            rows.iter()
                .find(|r| &r.tag == t && r.cell == cell)
                .and_then(|r| match &r.outcome {
                    Outcome::Measured(m) => ratio(m).map(|x| (t.as_str(), x)),
                    Outcome::NotRun(_) => None,
                })
        })
        .collect()
}

/// Judge the newest measured record of `cell` against the shrink-only floor.
pub(crate) fn ratchet(rows: &[Row], tags: &[String], cell: &str) -> Ratchet {
    let s = series(rows, tags, cell);
    let Some((&(tag, x), prior)) = s.split_last() else {
        return Ratchet::Unarmed { records: 0 };
    };
    if prior.len() < ARM_AFTER {
        return Ratchet::Unarmed { records: s.len() };
    }
    let ratios: Vec<f64> = prior.iter().map(|&(_, r)| r).collect();
    let floor = ratios
        .windows(ARM_AFTER)
        .map(median3)
        .fold(f64::NEG_INFINITY, f64::max);
    let (tag, ratio) = (tag.to_string(), x);
    if x < floor * (1.0 - TOLERANCE) {
        Ratchet::Red { tag, ratio, floor }
    } else {
        Ratchet::Green { tag, ratio, floor }
    }
}

#[cfg(test)]
#[path = "speed_ledger_tests.rs"]
mod tests;
