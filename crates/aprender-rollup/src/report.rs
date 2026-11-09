//! Human/machine text for a rollup plan (what `rollup-bot plan` prints).

use crate::{Reject, Selection};

fn reason(r: &Reject) -> String {
    match r {
        Reject::NotGreen => "checks not all green".to_string(),
        Reject::NotApproved => "not approved".to_string(),
        Reject::NoOptInLabel => format!("missing label {}", crate::OPT_IN_LABEL),
        Reject::TooLarge(n) => format!("{n} lines > {}", crate::MAX_LINES),
        Reject::ForbiddenFile(f) => format!("touches {f} (CI, contract or lockfile)"),
        Reject::FileOverlap { with, file } => format!("shares {file} with #{with}"),
        Reject::OverCap => format!("rollup already holds {}", crate::MAX_ENTRIES),
    }
}

/// One line for the rollup, then one `skip` line per rejected PR.
#[must_use]
pub fn plan_text(sel: &Selection) -> String {
    let picked: Vec<String> = sel.picked.iter().map(|n| format!("#{n}")).collect();
    let mut out = format!("rollup {} [{}]\n", sel.picked.len(), picked.join(" "));
    for (n, r) in &sel.rejected {
        out.push_str(&format!("skip #{n}: {}\n", reason(r)));
    }
    out
}

#[cfg(test)]
mod tests;
