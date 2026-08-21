//! Gate: PV-DUP-001 — duplicate contract stems.
//!
//! Contracts are addressed by STEM (`apr-cli-v1`), not by path: `depends_on` and
//! `assumes.from_contract` both name a bare stem. `contracts/` is a tree, so two
//! files in different directories can carry the same stem. When they do, any code
//! that collapses the corpus into a stem-keyed map has to pick one — and the pick
//! was previously made by `read_dir` order.
//!
//! That made a REQUIRED status check a function of filesystem walk order. It was
//! falsified by renaming a directory and touching no file content:
//! `mv contracts/aprender contracts/e-aprender` moved the composition gate from
//! `edges_broken: 0` to `edges_broken: 5`, flipping `pv lint` PASS -> FAIL.
//!
//! The fix has two halves, and BOTH are needed:
//!
//! 1. **Refuse, do not resolve.** A stem whose copies have DIVERGENT content is
//!    *ambiguous*. There is no defensible winner — the 546-line and the 357-line
//!    `apr-architecture-schema-v1` are not two renderings of one contract, they are
//!    two different contracts wearing one name. Ambiguous stems are excluded from
//!    the stem index entirely (see `composition_gate.rs`). A tie-break rule — even a
//!    deterministic one like "lexicographically greatest path wins" — would be the
//!    same defect with a stable seed, and would STILL move under a directory rename,
//!    because the directory name is part of the path that the rule sorts on.
//!
//!    Copies that are byte-identical are NOT ambiguous: every candidate resolves to
//!    the same contract, so the choice is invisible by construction.
//!
//! 2. **Report loudly, and ratchet.** Refusing silently would hide the corpus defect.
//!    Every divergent stem is listed in the gate output on every run, and the set is
//!    frozen against `scripts/contract_duplicate_stem_baseline.txt`. A stem that
//!    diverges but is not in the baseline FAILS the gate; a baseline entry that no
//!    longer diverges also FAILS, so the number can only go down.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::time::Instant;

use super::rules::RuleSeverity;

use super::finding::LintFinding;
use super::{GateDetail, GateResult};

/// Path of the ratchet baseline, relative to the project root.
pub const BASELINE_REL_PATH: &str = "scripts/contract_duplicate_stem_baseline.txt";

/// One contract stem claimed by more than one file, with divergent content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DuplicateStem {
    /// The colliding stem, e.g. `apr-architecture-schema-v1`.
    pub stem: String,
    /// Every path claiming the stem, sorted. Always at least 2 entries.
    pub paths: Vec<String>,
    /// Number of DISTINCT contents among those paths. Always at least 2.
    pub variants: usize,
}

/// Scan a contract tree for stems claimed by files with divergent content.
///
/// Returns one entry per ambiguous stem, sorted by stem. Byte-identical copies
/// are deliberately omitted: they are duplication, not ambiguity.
pub fn scan_duplicate_stems(dir: &Path) -> Vec<DuplicateStem> {
    let mut paths = Vec::new();
    super::gates::collect_yaml_files(dir, &mut paths);
    paths.sort();

    let mut by_stem: BTreeMap<String, Vec<std::path::PathBuf>> = BTreeMap::new();
    for path in paths {
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        by_stem.entry(stem.to_string()).or_default().push(path);
    }

    by_stem
        .into_iter()
        .filter(|(_, ps)| ps.len() > 1)
        .filter_map(|(stem, ps)| build_duplicate(&stem, &ps))
        .collect()
}

/// Build a `DuplicateStem` if the copies diverge; `None` if they are identical.
fn build_duplicate(stem: &str, paths: &[std::path::PathBuf]) -> Option<DuplicateStem> {
    let mut contents: BTreeSet<Vec<u8>> = BTreeSet::new();
    for p in paths {
        contents.insert(std::fs::read(p).unwrap_or_default());
    }
    if contents.len() < 2 {
        return None;
    }
    Some(DuplicateStem {
        stem: stem.to_string(),
        paths: paths.iter().map(|p| p.display().to_string()).collect(),
        variants: contents.len(),
    })
}

/// The set of stems that must NOT be resolved to a single contract.
pub fn ambiguous_stems(duplicates: &[DuplicateStem]) -> BTreeSet<String> {
    duplicates.iter().map(|d| d.stem.clone()).collect()
}

/// Read the ratchet baseline. A missing file means an EMPTY baseline, which makes
/// every divergent stem a hard error — the safe direction for a tree that has none.
pub fn read_baseline(project_root: &Path) -> BTreeSet<String> {
    let path = project_root.join(BASELINE_REL_PATH);
    let Ok(text) = std::fs::read_to_string(path) else {
        return BTreeSet::new();
    };
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(ToString::to_string)
        .collect()
}

/// Run PV-DUP-001. Fails on any divergent stem outside the baseline, and on any
/// baseline entry that no longer diverges (so the ratchet cannot be left slack).
pub(crate) fn run_duplicate_stem_gate(
    duplicates: &[DuplicateStem],
    baseline: &BTreeSet<String>,
) -> (GateResult, Vec<LintFinding>) {
    let start = Instant::now();
    let found: BTreeSet<String> = ambiguous_stems(duplicates);

    let unbaselined: Vec<String> = found.difference(baseline).cloned().collect();
    let stale: Vec<String> = baseline.difference(&found).cloned().collect();
    let passed = unbaselined.is_empty() && stale.is_empty();

    let mut findings = Vec::new();
    for stem in &unbaselined {
        let paths = duplicates
            .iter()
            .find(|d| &d.stem == stem)
            .map_or_else(String::new, |d| d.paths.join(", "));
        findings.push(
            LintFinding::new(
                "PV-DUP-001",
                RuleSeverity::Error,
                format!(
                    "Stem `{stem}` is claimed by multiple files with DIVERGENT content, \
                     so it cannot be resolved to one contract: {paths}"
                ),
                format!("contracts/{stem}.yaml"),
            )
            .with_stem(stem.clone()),
        );
    }
    for stem in &stale {
        findings.push(LintFinding::new(
            "PV-DUP-002",
            RuleSeverity::Error,
            format!(
                "Stem `{stem}` no longer diverges — remove it from {BASELINE_REL_PATH}. \
                 The ratchet only turns one way."
            ),
            BASELINE_REL_PATH.to_string(),
        ));
    }

    let result = GateResult {
        name: "duplicate-stems".into(),
        passed,
        skipped: false,
        duration_ms: u64::try_from(start.elapsed().as_millis()).unwrap_or(0),
        detail: GateDetail::DuplicateStems {
            divergent: duplicates.len(),
            baselined: found.intersection(baseline).count(),
            unbaselined,
            stale,
            divergent_stems: duplicates
                .iter()
                .map(|d| {
                    format!(
                        "{} [{} variants] {}",
                        d.stem,
                        d.variants,
                        d.paths.join(" | ")
                    )
                })
                .collect(),
        },
    };
    (result, findings)
}

#[cfg(test)]
#[path = "duplicate_stems_tests.rs"]
mod tests;
