//! REX-02 corpus builder: turns merged PRs and cargo-mutants listings into
//! corpus items. Reads git (and `gh` for the G-class green check); all
//! selection is seeded and ordered, so a rebuild from the same inputs yields
//! the same items and the same manifest.

use crate::corpus::{
    drop_comment_only_hunks, hunk_fingerprints, is_review_path, names_the_mutation, pick_balanced,
    sanitize_mutant_diff, sha256_hex, Class, Item, Sealed, Split,
};
use crate::stats::SplitMix64;
use serde::Deserialize;
use std::collections::BTreeSet;
use std::process::Command;

/// Target n per class (§2.2).
pub const PER_CLASS: usize = 50;
/// Candidate cap `[A]`: a diff over ~32k proxy tokens is dropped at
/// construction (release merge-backs, vendored drops). Anything kept is never
/// truncated; one that overflows a model's context is `NotRun{ContextOverflow}`.
pub const MAX_TOKENS: u64 = 32_000;

/// The subset of `gh pr list --json` the builder reads.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Pr {
    pub number: u64,
    pub title: String,
    pub merged_at: String,
    pub merge_commit: Option<Oid>,
    #[serde(default)]
    pub labels: Vec<Named>,
    #[serde(default)]
    pub closing_issues_references: Vec<IssueRef>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Oid {
    pub oid: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Named {
    pub name: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct IssueRef {
    pub number: u64,
}

/// The subset of a `cargo mutants --list --diff --json` row the builder reads.
#[derive(Debug, Clone, Deserialize)]
pub struct Mutant {
    pub name: String,
    pub file: String,
    pub diff: String,
    pub package: String,
}

/// A fix title (R class needs a fix PR, not a feature that closed an issue).
#[must_use]
pub fn is_fix_title(title: &str) -> bool {
    let t = title.to_lowercase();
    t.starts_with("fix")
        || t.starts_with("bug")
        || [
            "fix(",
            ": fix",
            "defect",
            "regression",
            "broken",
            "wrong",
            "crash",
            "panic",
        ]
        .iter()
        .any(|k| t.contains(k))
}

/// PR numbers named by a merged `Revert …` PR (`#N` in its title).
#[must_use]
pub fn reverted_prs(prs: &[Pr]) -> BTreeSet<u64> {
    prs.iter()
        .filter(|p| p.title.to_lowercase().starts_with("revert"))
        .flat_map(|p| {
            p.title
                .split('#')
                .skip(1)
                .filter_map(|s| {
                    let d: String = s.chars().take_while(char::is_ascii_digit).collect();
                    d.parse().ok()
                })
                .collect::<Vec<u64>>()
        })
        .collect()
}

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Review paths a commit touched.
fn review_paths(oid: &str) -> Vec<String> {
    let parent = format!("{oid}^");
    git(&["diff", "--name-only", &parent, oid])
        .unwrap_or_default()
        .lines()
        .filter(|p| is_review_path(p))
        .map(str::to_string)
        .collect()
}

/// `git diff -U3 from to -- paths`, or `None` when empty or over the cap.
fn diff_between(from: &str, to: &str, paths: &[String]) -> Option<String> {
    if paths.is_empty() {
        return None;
    }
    let mut args = vec!["diff", "--no-color", "--no-ext-diff", "-U3", from, to, "--"];
    args.extend(paths.iter().map(String::as_str));
    let d = git(&args)?;
    (!d.is_empty() && (d.len() as u64).div_ceil(4) <= MAX_TOKENS).then_some(d)
}

/// R candidates: fix PRs with a linked issue, fix hunks reverse-applied.
#[must_use]
pub fn r_candidates(prs: &[Pr]) -> Vec<(Item, String)> {
    prs.iter()
        .filter(|p| !p.closing_issues_references.is_empty() && is_fix_title(&p.title))
        .filter_map(|p| {
            let oid = &p.merge_commit.as_ref()?.oid;
            let paths = review_paths(oid);
            let diff = drop_comment_only_hunks(&diff_between(oid, &format!("{oid}^"), &paths)?);
            if diff.is_empty() {
                return None;
            }
            let issues: Vec<String> = p
                .closing_issues_references
                .iter()
                .map(|i| format!("#{}", i.number))
                .collect();
            let source = format!("pr#{} {oid} reversed; fixes {}", p.number, issues.join(","));
            let item = Item::new(format!("R-pr{}", p.number), Class::R, source, &diff);
            Some((item, diff))
        })
        .collect()
}

/// G candidates: merged before `cutoff`, not a fix, not reverted, no
/// `regression` label. The green check is applied afterwards ([`is_green`]).
#[must_use]
pub fn g_candidates(prs: &[Pr], cutoff: &str) -> Vec<(Item, String)> {
    let reverted = reverted_prs(prs);
    prs.iter()
        .filter(|p| p.merged_at.as_str() < cutoff)
        .filter(|p| !is_fix_title(&p.title) && !p.title.to_lowercase().starts_with("revert"))
        .filter(|p| !reverted.contains(&p.number))
        .filter(|p| !p.labels.iter().any(|l| l.name == "regression"))
        .filter_map(|p| {
            let oid = &p.merge_commit.as_ref()?.oid;
            let paths = review_paths(oid);
            let diff = diff_between(&format!("{oid}^"), oid, &paths)?;
            let source = format!("pr#{} {oid}", p.number);
            let item = Item::new(format!("G-pr{}", p.number), Class::G, source, &diff);
            Some((item, diff))
        })
        .collect()
}

/// Green clean-room: `ci / gate` and `workspace-test` both SUCCESS and no
/// check FAILED. Unknown (gh error) is not green.
#[must_use]
pub fn is_green(pr: u64) -> bool {
    let q = ".statusCheckRollup[]|\"\\(.name // .context)|\\(.conclusion // .state)\"";
    let Ok(out) = Command::new("gh")
        .args(["pr", "view", &pr.to_string(), "-R", "paiml/aprender"])
        .args(["--json", "statusCheckRollup", "-q", q])
        .output()
    else {
        return false;
    };
    out.status.success() && rollup_is_green(&String::from_utf8_lossy(&out.stdout))
}

/// [`is_green`] on the `name|conclusion` lines.
#[must_use]
pub fn rollup_is_green(lines: &str) -> bool {
    let ok = |n: &str| lines.lines().any(|l| l == format!("{n}|SUCCESS"));
    let failed = lines
        .lines()
        .any(|l| l.ends_with("|FAILURE") || l.ends_with("|TIMED_OUT") || l.ends_with("|ERROR"));
    ok("ci / gate") && ok("workspace-test") && !failed
}

/// P candidates: a seeded sample of mutants on review paths, one per file,
/// with every mutation marker stripped.
#[must_use]
pub fn p_candidates(mutants: &[Mutant], base: &str, n: usize, seed: u64) -> Vec<(Item, String)> {
    let mut ms: Vec<&Mutant> = mutants.iter().filter(|m| is_review_path(&m.file)).collect();
    ms.sort_by(|a, b| (&a.package, &a.name).cmp(&(&b.package, &b.name)));
    let mut rng = SplitMix64::new(seed);
    let keys: Vec<u64> = ms.iter().map(|_| rng.next_u64()).collect();
    let mut order: Vec<usize> = (0..ms.len()).collect();
    order.sort_by_key(|&i| keys[i]);
    let mut files = BTreeSet::new();
    let mut out = Vec::new();
    for i in order {
        let m = ms[i];
        if out.len() == n || !files.insert(m.file.clone()) {
            continue;
        }
        let diff = sanitize_mutant_diff(&m.diff, &m.file);
        if names_the_mutation(&diff) {
            continue;
        }
        let id = format!("P-{}", &sha256_hex(m.name.as_bytes())[..12]);
        let source = format!("cargo-mutants 27.0.0 @ {base}: {} :: {}", m.package, m.name);
        out.push((Item::new(id, Class::P, source, &diff), diff));
    }
    out
}

/// Keep the diffs of the items `pick_balanced` chose.
#[must_use]
pub fn choose(cands: Vec<(Item, String)>, n: usize, seed: u64) -> Vec<(Item, String)> {
    let items: Vec<Item> = cands.iter().map(|(i, _)| i.clone()).collect();
    let keep: BTreeSet<String> = pick_balanced(items, n, seed)
        .into_iter()
        .map(|i| i.id)
        .collect();
    cands
        .into_iter()
        .filter(|(i, _)| keep.contains(&i.id))
        .collect()
}

/// Sealed rows for the test split.
#[must_use]
pub fn seal(items: &[(Item, String)]) -> Vec<Sealed> {
    items
        .iter()
        .filter(|(i, _)| i.split == Split::Test)
        .map(|(i, d)| Sealed {
            id: i.id.clone(),
            diff_sha256: i.diff_sha256.clone(),
            hunks: hunk_fingerprints(d),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pr(n: u64, title: &str) -> Pr {
        Pr {
            number: n,
            title: title.into(),
            merged_at: "2026-09-01T00:00:00Z".into(),
            merge_commit: None,
            labels: vec![],
            closing_issues_references: vec![],
        }
    }

    #[test]
    fn fix_titles() {
        for t in [
            "fix(guard): x",
            "Fix x",
            "feat: y — regression in z",
            "cgp: wrong device",
        ] {
            assert!(is_fix_title(t), "{t}");
        }
        for t in ["feat(apr): add serve", "docs: prefix", "chore: bump"] {
            assert!(!is_fix_title(t), "{t}");
        }
    }

    #[test]
    fn reverts_name_their_targets() {
        let prs = vec![
            pr(9, "Revert \"feat: a (#1234)\" (#1240)"),
            pr(1, "feat #77"),
        ];
        assert_eq!(reverted_prs(&prs), BTreeSet::from([1234, 1240]));
    }

    #[test]
    fn green_needs_both_required_checks_and_no_failure() {
        let g = "ci / gate|SUCCESS\nworkspace-test|SUCCESS\nci / bench|SKIPPED\n";
        assert!(rollup_is_green(g));
        assert!(!rollup_is_green("ci / gate|SUCCESS\n"));
        assert!(!rollup_is_green(&format!("{g}mutants|FAILURE\n")));
        assert!(!rollup_is_green(""));
    }

    #[test]
    fn p_sample_is_seeded_one_per_file_and_marker_free() {
        let m = |name: &str, file: &str| {
            Mutant {
            name: name.into(),
            file: file.into(),
            package: "p".into(),
            diff: format!("--- {file}\n+++ replace {name}\n@@ -1,4 +1,4 @@\n a\n-b\n+c /* ~ changed by cargo-mutants ~ */\n d\n"),
        }
        };
        let ms = vec![
            m("a1", "src/a.rs"),
            m("a2", "src/a.rs"),
            m("b1", "src/b.rs"),
            m("t", "tests/t.rs"),
        ];
        let p = p_candidates(&ms, "base", 10, 4354);
        assert_eq!(p.len(), 2, "one per file, tests excluded");
        assert!(p
            .iter()
            .all(|(i, d)| !names_the_mutation(d) && i.class == Class::P));
        let ids = |v: &Vec<(Item, String)>| v.iter().map(|(i, _)| i.id.clone()).collect::<Vec<_>>();
        assert_eq!(ids(&p), ids(&p_candidates(&ms, "base", 10, 4354)));
    }
}
