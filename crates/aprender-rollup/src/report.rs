//! Human/machine text for a rollup plan (what `rollup-bot plan` prints).

use crate::{Config, Reject, Selection};

fn reason(cfg: &Config, r: &Reject) -> String {
    match r {
        Reject::NotGreen => "checks not all green".to_string(),
        Reject::NotApproved => "not approved".to_string(),
        Reject::NoOptInLabel => format!("missing label {}", cfg.label),
        Reject::TooLarge(n) => format!("{n} lines > {}", cfg.max_lines),
        Reject::ForbiddenFile(f) => format!("touches {f} (forbidden for this repo)"),
        Reject::FileOverlap { with, file } => format!("shares {file} with #{with}"),
        Reject::OverCap => format!("rollup already holds {}", cfg.max_entries),
    }
}

/// One line for the rollup, then one `skip` line per rejected PR.
#[must_use]
pub fn plan_text(cfg: &Config, sel: &Selection) -> String {
    let picked: Vec<String> = sel.picked.iter().map(|n| format!("#{n}")).collect();
    let mut out = format!("rollup {} [{}]\n", sel.picked.len(), picked.join(" "));
    for (n, r) in &sel.rejected {
        out.push_str(&format!("skip #{n}: {}\n", reason(cfg, r)));
    }
    out
}

#[cfg(test)]
mod tests;
