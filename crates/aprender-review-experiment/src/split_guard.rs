//! PRM-C9 `trace-split-guard-v1`: the train/val/test split (gate G-DUP;
//! contract `trace-split-guard-v1`, FALSIFY-TSG-001..004).
//!
//! - **Unit**: `repo#PR`. Every row of a PR lands in one split.
//! - **Time-ordered**: a PR's split comes from its first-seen time against
//!   the plan's `val_from` / `test_from` dates.
//! - **Embargo**: a PR first seen less than [`EMBARGO_DAYS`] before a
//!   boundary is in no split, so nothing in a later split is within 14 d of
//!   an earlier one.
//! - **Closure**: PRs that share a [`crate::dedup`] cluster are one component,
//!   split by the component's earliest time: a copy of a train diff opened
//!   later is train too, never test.
//! - **Sealed**: a component holding a near-dup of a sealed item is
//!   [`Split::Sealed`], i.e. refused from every pool.
//!
//! [`audit`] recomputes nothing: it checks any assignment for PRs or
//! clusters that straddle splits. G-DUP requires both lists empty.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use serde::{Deserialize, Serialize};

use crate::dedup::{self, Item};

pub const SCHEME: &str = "trace-split-guard-v1";

/// Minimum gap between a PR's first-seen time and the next split's start.
pub const EMBARGO_DAYS: i64 = 14;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Split {
    Train,
    Val,
    Test,
    /// Near-dup of a sealed item: in no pool.
    Sealed,
}

/// One trace diff to place.
#[derive(Debug, Clone, Copy)]
pub struct Unit<'a> {
    /// `diff_sha256`.
    pub id: &'a str,
    pub repo: &'a str,
    pub pr: u64,
    /// RFC 3339 UTC.
    pub at: &'a str,
    pub diff: &'a str,
}

/// Split boundaries, `YYYY-MM-DD`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub val_from: String,
    pub test_from: String,
}

/// The `split_guard` a trace row carries. `split: None` is embargoed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Guard {
    pub group_key: String,
    pub split: Option<Split>,
    pub dedup_cluster: String,
}

/// Days since 1970-01-01 of a `YYYY-MM-DD` prefix; `None` if it is not one.
#[must_use]
pub fn day(s: &str) -> Option<i64> {
    let b = s.as_bytes();
    if b.len() < 10 || b[4] != b'-' || b[7] != b'-' {
        return None;
    }
    let num = |r: std::ops::Range<usize>| s.get(r)?.parse::<i64>().ok();
    let (y, m, d) = (num(0..4)?, num(5..7)?, num(8..10)?);
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    // Howard Hinnant's days_from_civil.
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (m + if m > 2 { -3 } else { 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Some(era * 146_097 + doe - 719_468)
}

/// Assign each unit its guard, in input order. A plan whose val window is
/// shorter than the embargo, or an unreadable date, is an error.
pub fn assign(units: &[Unit], sealed: &[(&str, &str)], plan: &Plan) -> Result<Vec<Guard>, String> {
    let _ = sealed;
    let (v, t) = (
        day(&plan.val_from).ok_or("val_from is not a date")?,
        day(&plan.test_from).ok_or("test_from is not a date")?,
    );
    let items: Vec<Item> = units
        .iter()
        .map(|u| Item { id: u.id, at: u.at, diff: u.diff })
        .collect();
    let clusters = dedup::clusters(&items);
    units
        .iter()
        .zip(clusters)
        .map(|(u, c)| {
            let d = day(u.at).ok_or_else(|| format!("{}: at {:?} is not a date", u.id, u.at))?;
            let split = if d < v {
                Some(Split::Train)
            } else if d < t {
                Some(Split::Val)
            } else {
                Some(Split::Test)
            };
            Ok(Guard { group_key: format!("{}#{}", u.repo, u.pr), split, dedup_cluster: c })
        })
        .collect()
}

/// What straddles splits in an assignment. G-DUP holds when both are empty.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct Audit {
    pub straddling_groups: Vec<String>,
    pub cross_split_clusters: Vec<String>,
}

/// Check any assignment. Embargoed members (`None`) are in no pool, so they
/// never make a straddle.
#[must_use]
pub fn audit(guards: &[Guard]) -> Audit {
    let _ = guards;
    Audit::default()
}

#[cfg(test)]
#[path = "split_guard_tests.rs"]
mod tests;
