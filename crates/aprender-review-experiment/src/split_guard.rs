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

/// The `YYYY-MM-DD` of a [`day`] number (Hinnant's civil_from_days).
#[must_use]
pub fn date(days: i64) -> String {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}")
}

/// Assign each unit its guard, in input order. A plan whose val window is
/// shorter than the embargo, or an unreadable date, is an error.
pub fn assign(units: &[Unit], sealed: &[(&str, &str)], plan: &Plan) -> Result<Vec<Guard>, String> {
    let (v, t) = (
        day(&plan.val_from).ok_or("val_from is not a date")?,
        day(&plan.test_from).ok_or("test_from is not a date")?,
    );
    if t - v <= EMBARGO_DAYS {
        return Err(format!(
            "val window {} d is not longer than the {EMBARGO_DAYS} d embargo",
            t - v
        ));
    }
    let days = units
        .iter()
        .map(|u| day(u.at).ok_or_else(|| format!("{}: at {:?} is not a date", u.id, u.at)))
        .collect::<Result<Vec<i64>, String>>()?;
    // Sealed items sort before every unit (empty `at`), so they name their clusters.
    let items: Vec<Item> = sealed
        .iter()
        .map(|&(id, diff)| Item { id, at: "", diff })
        .chain(units.iter().map(|u| Item {
            id: u.id,
            at: u.at,
            diff: u.diff,
        }))
        .collect();
    let all = dedup::clusters(&items);
    let (sealed_ids, clusters) = all.split_at(sealed.len());
    let sealed_ids: BTreeSet<&String> = sealed_ids.iter().collect();
    // Components: units joined by a shared PR or a shared cluster.
    let keys: Vec<String> = units
        .iter()
        .map(|u| format!("{}#{}", u.repo, u.pr))
        .collect();
    let mut parent: Vec<usize> = (0..units.len()).collect();
    let mut first: HashMap<&str, usize> = HashMap::new();
    for i in 0..units.len() {
        for k in [keys[i].as_str(), clusters[i].as_str()] {
            let j = *first.entry(k).or_insert(i);
            let (ri, rj) = (root(&mut parent, i), root(&mut parent, j));
            parent[ri.max(rj)] = ri.min(rj);
        }
    }
    let mut comp: BTreeMap<usize, (i64, bool)> = BTreeMap::new();
    for i in 0..units.len() {
        let r = root(&mut parent, i);
        let e = comp.entry(r).or_insert((days[i], false));
        e.0 = e.0.min(days[i]);
        e.1 |= sealed_ids.contains(&clusters[i]);
    }
    Ok((0..units.len())
        .map(|i| {
            let (d, is_sealed) = comp[&root(&mut parent, i)];
            let split = if is_sealed {
                Some(Split::Sealed)
            } else if d < v - EMBARGO_DAYS {
                Some(Split::Train)
            } else if d < v {
                None
            } else if d < t - EMBARGO_DAYS {
                Some(Split::Val)
            } else if d < t {
                None
            } else {
                Some(Split::Test)
            };
            Guard {
                group_key: keys[i].clone(),
                split,
                dedup_cluster: clusters[i].clone(),
            }
        })
        .collect())
}

fn root(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
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
    let mut by_group: BTreeMap<&str, BTreeSet<Split>> = BTreeMap::new();
    let mut by_cluster: BTreeMap<&str, BTreeSet<Split>> = BTreeMap::new();
    for g in guards {
        if let Some(s) = g.split {
            by_group.entry(&g.group_key).or_default().insert(s);
            by_cluster.entry(&g.dedup_cluster).or_default().insert(s);
        }
    }
    let straddles = |m: BTreeMap<&str, BTreeSet<Split>>| -> Vec<String> {
        m.into_iter()
            .filter(|(_, s)| s.len() > 1)
            .map(|(k, _)| k.to_owned())
            .collect()
    };
    Audit {
        straddling_groups: straddles(by_group),
        cross_split_clusters: straddles(by_cluster),
    }
}

#[cfg(test)]
#[path = "split_guard_tests.rs"]
mod tests;
