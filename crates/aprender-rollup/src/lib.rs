//! Rollup bot core (T-rollup, refs #5394): pure selection and bisection, no I/O.
//!
//! The `gh` layer feeds [`Candidate`]s in; [`select`] picks a rollup; on a red
//! merge-group run [`split`] halves it. The bot closes a constituent only after the
//! rollup commit is on main, with the SHA posted; never on red (see [`Outcome`]).

pub mod build;
pub mod close;
pub mod gh_json;
pub mod mg;
pub mod report;

/// Largest PR (additions + deletions, from `gh`) that may join a rollup.
pub const MAX_LINES: u32 = 200;
/// Most constituents in one rollup (operator: b = 3, only with bisection, until p_red is measured).
pub const MAX_ENTRIES: usize = 3;
/// Label a PR owner sets to opt in.
pub const OPT_IN_LABEL: &str = "rollup-ok";
/// Label a capacity PR (runner seats, host readiness) carries to take the next free slot (infra#2247).
pub const CAPACITY_LABEL: &str = "capacity";
/// Declared capacity paths (infra-66, mirrors infra `fleet/capacity-paths.txt`): a PR is capacity only if
/// EVERY changed path matches one of these globs (`**` crosses `/`, `*` stays inside one segment).
pub const CAPACITY_PATHS: &[&str] = &[
    "machines/clean-room/runner/**",
    "machines/intel/forjar-ci.yaml",
    "machines/framework16/forjar.yaml",
    "machines/framework16/Makefile",
    "machines/gx10/forjar-*.yaml",
    "machines/t2*/forjar.yaml",
    "machines/fleet-hosts/**",
];

/// Glob match for [`CAPACITY_PATHS`]: `**` matches any run of characters, `*` any run without `/`.
#[must_use]
pub fn glob_match(pat: &str, path: &str) -> bool {
    fn go(p: &[u8], s: &[u8]) -> bool {
        match p.split_first() {
            None => s.is_empty(),
            Some((b'*', rest)) if rest.first() == Some(&b'*') => {
                let rest = &rest[1..];
                (0..=s.len()).any(|i| go(rest, &s[i..]))
            }
            Some((b'*', rest)) => {
                let n = s.iter().position(|&b| b == b'/').unwrap_or(s.len());
                (0..=n).any(|i| go(rest, &s[i..]))
            }
            Some((c, rest)) => s.first() == Some(c) && go(rest, &s[1..]),
        }
    }
    go(pat.as_bytes(), path.as_bytes())
}

fn in_capacity_paths(path: &str) -> bool {
    CAPACITY_PATHS.iter().any(|p| glob_match(p, path))
}

/// One open PR as read from GitHub.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    pub number: u64,
    pub additions: u32,
    pub deletions: u32,
    pub green: bool,
    pub approved: bool,
    pub labels: Vec<String>,
    pub files: Vec<String>,
    /// Head commit the checks and approval were read at.
    pub head: String,
}

/// Why a PR stays out of a rollup; posted as the ejection reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Reject {
    NotGreen,
    NotApproved,
    NoOptInLabel,
    TooLarge(u32),
    ForbiddenFile(String),
    FileOverlap {
        with: u64,
        file: String,
    },
    /// Eligible, but the rollup already holds [`MAX_ENTRIES`].
    OverCap,
    /// Carries [`CAPACITY_LABEL`] but changes a path outside [`CAPACITY_PATHS`] (the label is refused).
    NotCapacityPaths(String),
}

/// Files a rollup never carries: CI workflows, contracts, the lockfile.
#[must_use]
pub fn is_forbidden(path: &str) -> bool {
    path.starts_with(".github/workflows/") || path.starts_with("contracts/") || path == "Cargo.lock"
}

/// First reason `c` is ineligible on its own, if any.
#[must_use]
pub fn check(c: &Candidate) -> Option<Reject> {
    if !c.green {
        return Some(Reject::NotGreen);
    }
    if !c.approved {
        return Some(Reject::NotApproved);
    }
    if !c.labels.iter().any(|l| l == OPT_IN_LABEL) {
        return Some(Reject::NoOptInLabel);
    }
    let lines = c.additions.saturating_add(c.deletions);
    if lines > MAX_LINES {
        return Some(Reject::TooLarge(lines));
    }
    c.files
        .iter()
        .find(|f| is_forbidden(f))
        .map(|f| Reject::ForbiddenFile(f.clone()))
}

/// A capacity PR: the label AND every changed path in [`CAPACITY_PATHS`] (no files = not capacity).
#[must_use]
pub fn is_capacity(c: &Candidate) -> bool {
    c.labels.iter().any(|l| l == CAPACITY_LABEL)
        && !c.files.is_empty()
        && c.files.iter().all(|f| in_capacity_paths(f))
}

/// A chosen rollup plus every PR left out and why.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Selection {
    pub picked: Vec<u64>,
    /// Head SHA of each pick (parallel to `picked`), re-checked when the rollup is built.
    pub heads: Vec<String>,
    pub rejected: Vec<(u64, Reject)>,
}

/// Pick up to [`MAX_ENTRIES`] eligible PRs, capacity PRs first then lowest number, no shared file.
#[must_use]
pub fn select(cands: &[Candidate]) -> Selection {
    let mut sorted: Vec<&Candidate> = cands.iter().collect();
    sorted.sort_by_key(|c| (!is_capacity(c), c.number));
    let mut sel = Selection::default();
    let mut taken: Vec<(&Candidate, u64)> = Vec::new();
    for c in sorted {
        if c.labels.iter().any(|l| l == CAPACITY_LABEL) && !is_capacity(c) {
            let other = c
                .files
                .iter()
                .find(|f| !in_capacity_paths(f))
                .cloned()
                .unwrap_or_default();
            sel.rejected
                .push((c.number, Reject::NotCapacityPaths(other)));
            continue;
        }
        if let Some(r) = check(c) {
            sel.rejected.push((c.number, r));
            continue;
        }
        let clash = taken.iter().find_map(|(t, n)| {
            c.files
                .iter()
                .find(|f| t.files.contains(f))
                .map(|f| (*n, f.clone()))
        });
        if let Some((with, file)) = clash {
            sel.rejected
                .push((c.number, Reject::FileOverlap { with, file }));
        } else if taken.len() < MAX_ENTRIES {
            sel.picked.push(c.number);
            sel.heads.push(c.head.clone());
            taken.push((c, c.number));
        } else {
            sel.rejected.push((c.number, Reject::OverCap));
        }
    }
    sel
}

/// Halve a red rollup for bisection. A single entry cannot be split.
#[must_use]
pub fn split(picked: &[u64]) -> Option<(Vec<u64>, Vec<u64>)> {
    if picked.len() < 2 {
        return None;
    }
    let (a, b) = picked.split_at(picked.len() / 2);
    Some((a.to_vec(), b.to_vec()))
}

/// What the bot may do with one constituent after a merge-group run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// Rollup commit is on main: close with the SHA posted.
    Close { sha: String },
    /// Red or not yet on main: leave open and say why.
    KeepOpen { reason: String },
}

/// Close only when the run is green and the rollup commit is on main.
#[must_use]
pub fn outcome(green: bool, sha_on_main: Option<&str>) -> Outcome {
    match (green, sha_on_main) {
        (true, Some(sha)) if !sha.is_empty() => Outcome::Close {
            sha: sha.to_string(),
        },
        (false, _) => Outcome::KeepOpen {
            reason: "rollup run red".to_string(),
        },
        _ => Outcome::KeepOpen {
            reason: "rollup commit not on main".to_string(),
        },
    }
}

#[cfg(test)]
mod tests;
