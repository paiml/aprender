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
    let _ = (events, now, WINDOW_DAYS, date(0), day(""));
    Ok(Vec::new())
}

/// Gold labels: matured outcomes, then the latest HRQ ruling per round. An
/// outcome that is neither pending nor matured, or a ruling with no ruler or
/// date, is an error.
pub fn gold(resolved: &[Resolved], rulings: &[Ruling]) -> Result<Vec<Label>, String> {
    let _ = (resolved, rulings, day_of("", ""));
    Ok(Vec::new())
}

/// Outcomes whose window closed in `[from, to)` (`YYYY-MM-DD`).
#[must_use]
pub fn matured_between(resolved: &[Resolved], from: &str, to: &str) -> usize {
    let _ = (resolved, from, to);
    0
}

#[cfg(test)]
#[path = "outcome_join_tests.rs"]
mod tests;
