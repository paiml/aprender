//! Merge-group result check, bisection on red, and the close plan (T-rollup, #5394).
//!
//! Pure: the caller runs `gh run list --json status,conclusion` and passes the text, and
//! supplies the "test this subset" step to [`bisect`]. A constituent is closed only when
//! its subset went green and the rollup SHA is on main ([`crate::outcome`]).

use crate::{close, outcome};
use serde_json::Value;

/// State of the merge-group runs for one rollup (or one half of it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Green,
    Red,
    Pending,
}

/// Fold `gh run list --json status,conclusion,headSha` (only runs of `head` count) into one verdict. Any failed run is red;
/// no runs, or any unfinished run with none failed, is pending (never green).
///
/// # Errors
/// Returns the parse error text (a failed read is not_measured, L23).
pub fn parse_verdict(json: &str, head: &str) -> Result<Verdict, String> {
    let v: Value = serde_json::from_str(json).map_err(|e| e.to_string())?;
    let runs: Vec<&Value> = v
        .as_array()
        .ok_or("expected a JSON array")?
        .iter()
        .filter(|r| r["headSha"].as_str() == Some(head))
        .collect();
    // Green needs every run finished and at least one real success: all-skipped is not a pass.
    let (mut unfinished, mut succeeded) = (false, false);
    for r in &runs {
        match (r["status"].as_str(), r["conclusion"].as_str()) {
            (Some("completed"), Some("success")) => succeeded = true,
            (Some("completed"), Some("neutral" | "skipped")) => {}
            (Some("completed"), Some(_)) => return Ok(Verdict::Red),
            _ => unfinished = true,
        }
    }
    Ok(if succeeded && !unfinished {
        Verdict::Green
    } else {
        Verdict::Pending
    })
}

/// `gh` args listing the merge-group runs for a rollup branch.
#[must_use]
pub fn run_list_args(branch: &str) -> Vec<String> {
    [
        "run",
        "list",
        "--event",
        "merge_group",
        "--branch",
        branch,
        "--json",
        "status,conclusion,headSha",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect()
}

/// `git` args: is `sha` an ancestor of `origin/main` (i.e. on main)?
#[must_use]
pub fn ancestor_args(sha: &str) -> Vec<String> {
    ["merge-base", "--is-ancestor", sha, "origin/main"]
        .iter()
        .map(|s| (*s).to_string())
        .collect()
}

/// Result of bisecting a rollup: who may land, who broke it, who is still unknown.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Bisect {
    pub good: Vec<u64>,
    pub culprits: Vec<u64>,
    pub undecided: Vec<u64>,
}

/// Halve a red set until each culprit is isolated. `test` runs a subset through the merge
/// group. A green subset is good; a red single entry is a culprit; pending stays undecided.
pub fn bisect(picked: &[u64], test: &mut dyn FnMut(&[u64]) -> Verdict) -> Bisect {
    let mut out = Bisect::default();
    walk(picked, test, &mut out);
    out.good.sort_unstable();
    out.culprits.sort_unstable();
    out.undecided.sort_unstable();
    out
}

fn walk(set: &[u64], test: &mut dyn FnMut(&[u64]) -> Verdict, out: &mut Bisect) {
    if set.is_empty() {
        return;
    }
    match test(set) {
        Verdict::Green => out.good.extend_from_slice(set),
        Verdict::Pending => out.undecided.extend_from_slice(set),
        Verdict::Red => match crate::split(set) {
            None => out.culprits.extend_from_slice(set),
            Some((a, b)) => {
                walk(&a, test, out);
                walk(&b, test, out);
            }
        },
    }
}

/// One `gh` action per constituent. Good ones close with the SHA, only when it is on main;
/// culprits and undecided ones get a reasoned comment and stay open.
#[must_use]
pub fn close_plan(b: &Bisect, sha_on_main: Option<&str>) -> Vec<(u64, Vec<String>)> {
    let mut plan = Vec::new();
    for n in &b.good {
        plan.push((*n, close::gh_args(*n, &outcome(true, sha_on_main))));
    }
    for n in &b.culprits {
        plan.push((*n, close::gh_args(*n, &outcome(false, sha_on_main))));
    }
    for n in &b.undecided {
        let o = crate::Outcome::KeepOpen {
            reason: "rollup run not finished".to_string(),
        };
        plan.push((*n, close::gh_args(*n, &o)));
    }
    plan
}

#[cfg(test)]
mod tests;
