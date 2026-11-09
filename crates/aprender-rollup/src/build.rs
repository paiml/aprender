//! Git steps that assemble a rollup branch from picked PRs (pure; `main.rs` runs them).

/// Branch name for a rollup of `picked` (lowest PR number first, so it is stable).
#[must_use]
pub fn branch_name(picked: &[u64]) -> String {
    let first = picked.iter().min().copied().unwrap_or(0);
    format!("rollup/{first}-{}", picked.len())
}

/// Commit message: constituents are named so the bot can close them after the SHA lands.
#[must_use]
pub fn message(picked: &[u64]) -> String {
    let list: Vec<String> = picked.iter().map(|n| format!("#{n}")).collect();
    format!("Rollup of {} PRs: {}", picked.len(), list.join(" "))
}

/// Ordered git argument lists: branch from `base`, then fetch and merge each PR head.
#[must_use]
pub fn steps(base: &str, picked: &[u64]) -> Vec<Vec<String>> {
    let v = |a: &[&str]| a.iter().map(|s| (*s).to_string()).collect::<Vec<_>>();
    let mut out = vec![
        v(&["fetch", "origin", base]),
        v(&[
            "checkout",
            "-B",
            &branch_name(picked),
            &format!("origin/{base}"),
        ]),
    ];
    for n in picked {
        out.push(v(&["fetch", "origin", &format!("pull/{n}/head")]));
        out.push(v(&[
            "merge",
            "--no-ff",
            "-m",
            &format!("Merge #{n}"),
            "FETCH_HEAD",
        ]));
    }
    out
}

/// Print the commit the last `fetch` brought in, to compare with the head judged at selection.
#[must_use]
pub fn head_step() -> Vec<String> {
    vec!["rev-parse".to_string(), "FETCH_HEAD".to_string()]
}

/// Recovery step after a merge conflict: the PR is ejected and stays open.
#[must_use]
pub fn abort_step() -> Vec<String> {
    vec!["merge".to_string(), "--abort".to_string()]
}

#[cfg(test)]
mod tests;
