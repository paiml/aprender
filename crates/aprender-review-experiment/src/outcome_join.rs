//! PRM-C10 `trace-outcome-join-v1`: PR outcomes and HRQ rulings as gold
//! (gate G-MAT; contract `trace-outcome-join-v1`, FALSIFY-TOJ-001..005).
//!
//! - **Outcome**: from a PR's merge / close / revert / escape events. A revert
//!   at most [`WINDOW_DAYS`] after the merge is `reverted_le14d`; a later
//!   revert or any escape is `regression_escape`.
//! - **Maturity**: an outcome is gold only once its window has closed,
//!   [`WINDOW_DAYS`] UTC dates after the merge (or close). Until then it reads
//!   `pending`, and pending is never a negative.
//! - **HRQ rulings** are gold when made; the latest ruling on a round wins.
//! - [`gold`] emits rows whose `label_source` is a [`crate::pool::GOLD_SOURCES`]
//!   value; [`matured_between`] is the weekly matured count.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::ledger::Verdict;
use crate::split_guard::{date, day};

pub const SCHEME: &str = "trace-outcome-join-v1";

/// The outcome window, from the merge.
pub const WINDOW_DAYS: i64 = 14;

/// The v3 lifecycle `outcome`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    Pending,
    Merged,
    RevertedLe14d,
    RegressionEscape,
    ClosedUnmerged,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    Merged,
    Closed,
    Reverted,
    /// A regression traced to the PR.
    Escape,
}

/// One PR event, as the join collects it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Event {
    pub repo: String,
    pub pr: u64,
    pub kind: EventKind,
    /// RFC 3339 UTC.
    pub at: String,
}

/// A PR's outcome as of `now`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Resolved {
    /// `repo#PR`.
    pub group_key: String,
    pub outcome: Outcome,
    /// `YYYY-MM-DD` the window closed; `None` while pending.
    pub outcome_matured_at: Option<String>,
}

/// A human ruling on one quorum round.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ruling {
    pub repo: String,
    pub pr: u64,
    pub head: String,
    pub verdict: Verdict,
    pub ruled_by: String,
    /// RFC 3339 UTC.
    pub at: String,
}

/// One gold label.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "label_source", rename_all = "snake_case", deny_unknown_fields)]
pub enum Label {
    Outcome {
        group_key: String,
        outcome: Outcome,
        outcome_matured_at: String,
    },
    HrqRuling {
        group_key: String,
        head: String,
        verdict: Verdict,
        ruled_by: String,
        at: String,
    },
}

fn day_of(what: &str, at: &str) -> Result<i64, String> {
    day(at).ok_or_else(|| format!("{what}: at {at:?} is not a date"))
}

/// Each PR's outcome as of `now`, one row per `repo#PR`, sorted by key.
/// A revert or escape with no merge, or an unreadable date, is an error.
pub fn resolve(events: &[Event], now: &str) -> Result<Vec<Resolved>, String> {
    let today = day_of("now", now)?;
    let mut by_pr: BTreeMap<String, Vec<(EventKind, i64)>> = BTreeMap::new();
    for e in events {
        let key = format!("{}#{}", e.repo, e.pr);
        let d = day_of(&key, &e.at)?;
        by_pr.entry(key).or_default().push((e.kind, d));
    }
    by_pr
        .into_iter()
        .map(|(key, evs)| {
            let first = |k: EventKind| evs.iter().filter(|e| e.0 == k).map(|e| e.1).min();
            let (outcome, from) = if let Some(m) = first(EventKind::Merged) {
                let late = |k| evs.iter().any(|&(ek, d)| ek == k && d >= m);
                let outcome = match first(EventKind::Reverted).filter(|&r| r >= m) {
                    Some(r) if r - m <= WINDOW_DAYS => Outcome::RevertedLe14d,
                    Some(_) => Outcome::RegressionEscape,
                    None if late(EventKind::Escape) => Outcome::RegressionEscape,
                    None => Outcome::Merged,
                };
                (outcome, m)
            } else if let Some(c) = first(EventKind::Closed) {
                if evs
                    .iter()
                    .any(|e| matches!(e.0, EventKind::Reverted | EventKind::Escape))
                {
                    return Err(format!("{key}: a revert or escape of an unmerged PR"));
                }
                (Outcome::ClosedUnmerged, c)
            } else {
                return Err(format!("{key}: a revert or escape with no merge"));
            };
            let matured = today - from >= WINDOW_DAYS;
            Ok(Resolved {
                group_key: key,
                outcome: if matured { outcome } else { Outcome::Pending },
                outcome_matured_at: matured.then(|| date(from + WINDOW_DAYS)),
            })
        })
        .collect()
}

/// Gold labels: matured outcomes, then the latest HRQ ruling per round. An
/// outcome that is neither pending nor matured, or a ruling with no ruler or
/// date, is an error.
pub fn gold(resolved: &[Resolved], rulings: &[Ruling]) -> Result<Vec<Label>, String> {
    let mut out = Vec::new();
    for r in resolved {
        match (&r.outcome, &r.outcome_matured_at) {
            (Outcome::Pending, _) => {}
            (&outcome, Some(m)) => out.push(Label::Outcome {
                group_key: r.group_key.clone(),
                outcome,
                outcome_matured_at: m.clone(),
            }),
            (o, None) => return Err(format!("{}: {o:?} before its window closed", r.group_key)),
        }
    }
    let mut latest: BTreeMap<(String, &str), &Ruling> = BTreeMap::new();
    for r in rulings {
        let key = format!("{}#{}", r.repo, r.pr);
        if r.ruled_by.trim().is_empty() {
            return Err(format!("{key}@{}: ruling has no ruler", r.head));
        }
        day_of(&key, &r.at)?;
        let e = latest.entry((key, &r.head)).or_insert(r);
        if r.at > e.at {
            *e = r;
        }
    }
    out.extend(
        latest
            .into_iter()
            .map(|((group_key, head), r)| Label::HrqRuling {
                group_key,
                head: head.to_owned(),
                verdict: r.verdict,
                ruled_by: r.ruled_by.clone(),
                at: r.at.clone(),
            }),
    );
    Ok(out)
}

/// Outcomes whose window closed in `[from, to)` (`YYYY-MM-DD`).
#[must_use]
pub fn matured_between(resolved: &[Resolved], from: &str, to: &str) -> usize {
    resolved
        .iter()
        .filter_map(|r| r.outcome_matured_at.as_deref())
        .filter(|m| *m >= from && *m < to)
        .count()
}

#[cfg(test)]
#[path = "outcome_join_tests.rs"]
mod tests;
