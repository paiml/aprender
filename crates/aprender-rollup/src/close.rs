//! `gh` argument lists for the only two things the bot does to a constituent PR.

use crate::Outcome;

/// Comment-and-close args for a constituent, or a reasoned comment if it stays open.
/// Closing needs a non-empty SHA on main; `Outcome` already guarantees that.
#[must_use]
pub fn gh_args(pr: u64, outcome: &Outcome) -> Vec<String> {
    let n = pr.to_string();
    match outcome {
        Outcome::Close { sha } => {
            let body = format!("Landed in rollup commit {sha} on main.");
            ["pr", "close", &n, "--comment", &body]
                .iter()
                .map(|s| (*s).to_string())
                .collect()
        }
        Outcome::KeepOpen { reason } => {
            let body = format!("Ejected from rollup, left open: {reason}");
            ["pr", "comment", &n, "--body", &body]
                .iter()
                .map(|s| (*s).to_string())
                .collect()
        }
    }
}

#[cfg(test)]
mod tests;
